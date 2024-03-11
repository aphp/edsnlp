import os
from abc import ABCMeta
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import safetensors.torch
import torch
from spacy.tokens import Doc

from edsnlp.pipes.base import BaseComponent
from edsnlp.utils.collections import batch_compress_dict, batchify, decompress_dict

BatchInput = TypeVar("BatchInput", bound=Dict[str, Any])
BatchOutput = TypeVar("BatchOutput", bound=Dict[str, Any])
Scorer = Callable[[Sequence[Tuple[Doc, Doc]]], Union[float, Dict[str, Any]]]

ALL_CACHES = object()
_caches = {}


class CacheEnum(str, Enum):
    preprocess = "preprocess"
    collate = "collate"
    forward = "forward"


def hash_batch(batch):
    if isinstance(batch, list):
        return hash(tuple(id(item) for item in batch))
    elif not isinstance(batch, dict):
        return id(batch)
    if "__batch_hash__" in batch:
        return batch["__batch_hash__"]
    batch_hash = hash((tuple(batch.keys()), tuple(map(hash_batch, batch.values()))))
    batch["__batch_hash__"] = batch_hash
    return batch_hash


def cached_preprocess(fn):
    @wraps(fn)
    def wrapped(self: "TorchComponent", doc: Doc):
        if self._current_cache_id is None:
            return fn(self, doc)
        cache_key = (
            "preprocess",
            f"{type(self).__name__}<{id(self)}>",
            id(doc),
        )
        cache = _caches[self._current_cache_id]
        if cache_key in cache:
            return cache[cache_key]
        res = fn(self, doc)
        cache[cache_key] = res
        return res

    return wrapped


def cached_preprocess_supervised(fn):
    @wraps(fn)
    def wrapped(self: "TorchComponent", doc: Doc):
        if self._current_cache_id is None:
            return fn(self, doc)
        cache_key = (
            "preprocess_supervised",
            f"{type(self).__name__}<{id(self)}>",
            id(doc),
        )
        cache = _caches[self._current_cache_id]
        if cache_key in cache:
            return cache[cache_key]
        res = fn(self, doc)
        cache[cache_key] = res
        return res

    return wrapped


def cached_collate(fn):
    @wraps(fn)
    def wrapped(self: "TorchComponent", batch: Dict):
        if self._current_cache_id is None:
            return fn(self, batch)
        batch_hash = hash_batch(batch)
        cache_key = (
            "collate",
            f"{type(self).__name__}<{id(self)}>",
            batch_hash,
        )
        cache = _caches[self._current_cache_id]
        if cache_key in cache:
            return cache[cache_key]
        res = fn(self, batch)
        res["__batch_hash__"] = batch_hash
        cache[cache_key] = res
        return res

    return wrapped


def cached_forward(fn):
    @wraps(fn)
    def wrapped(self: "TorchComponent", batch):
        # Convert args and kwargs to a dictionary matching fn signature
        if self._current_cache_id is None:
            return fn(self, batch)
        cache_key = (
            "forward",
            f"{type(self).__name__}<{id(self)}>",
            hash_batch(batch),
        )
        cache = _caches[self._current_cache_id]
        if cache_key in cache:
            return cache[cache_key]
        res = fn(self, batch)
        cache[cache_key] = res
        return res

    return wrapped


def cached_batch_to_device(fn):
    @wraps(fn)
    def wrapped(self: "TorchComponent", batch, device):
        # Convert args and kwargs to a dictionary matching fn signature
        if self._current_cache_id is None:
            return fn(self, batch, device)
        cache_key = (
            "batch_to_device",
            f"{type(self).__name__}<{id(self)}>",
            hash_batch(batch),
        )
        cache = _caches[self._current_cache_id]
        if cache_key in cache:
            return cache[cache_key]
        res = fn(self, batch, device)
        cache[cache_key] = res
        return res

    return wrapped


class TorchComponentMeta(ABCMeta):
    def __new__(mcs, name, bases, class_dict):
        if "preprocess" in class_dict:
            class_dict["preprocess"] = cached_preprocess(class_dict["preprocess"])
        if "preprocess_supervised" in class_dict:
            class_dict["preprocess_supervised"] = cached_preprocess_supervised(
                class_dict["preprocess_supervised"]
            )
        if "collate" in class_dict:
            class_dict["collate"] = cached_collate(class_dict["collate"])
        if "batch_to_device" in class_dict:
            class_dict["batch_to_device"] = cached_batch_to_device(
                class_dict["batch_to_device"]
            )
        if "forward" in class_dict:
            class_dict["forward"] = cached_forward(class_dict["forward"])

        return super().__new__(mcs, name, bases, class_dict)


class TorchComponent(
    BaseComponent,
    torch.nn.Module,
    Generic[BatchOutput, BatchInput],
    metaclass=TorchComponentMeta,
):
    """
    A TorchComponent is a Component that can be trained and inherits `torch.nn.Module`.
    You can use it either as a torch module inside a more complex neural network, or as
    a standalone component in a [Pipeline][edsnlp.core.pipeline.Pipeline].

    In addition to the methods of a torch module, a TorchComponent adds a few methods to
    handle preprocessing and collating features, as well as caching intermediate results
    for components that share a common subcomponent.
    """

    call_super_init = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_cache_id = None

    def enable_cache(self, cache_id="default"):
        self._current_cache_id = cache_id
        _caches.setdefault(cache_id, {})
        for name, component in self.named_component_children():
            if hasattr(component, "enable_cache"):
                component.enable_cache(cache_id)

    def disable_cache(self, cache_id=ALL_CACHES):
        if cache_id is ALL_CACHES:
            _caches.clear()
        else:
            if cache_id in _caches:
                del _caches[cache_id]
        self._current_cache_id = None
        for name, component in self.named_component_children():
            if hasattr(component, "disable_cache"):
                component.disable_cache(cache_id)

    @property
    def device(self):
        return next((p.device for p in self.parameters()), torch.device("cpu"))

    def named_component_children(self):
        for name, module in self.named_children():
            if isinstance(module, TorchComponent):
                yield name, module

    def named_component_modules(self):
        for name, module in self.named_modules():
            if isinstance(module, TorchComponent):
                yield name, module

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]):
        """
        This method completes the attributes of the component, by looking at some
        documents. It is especially useful to build vocabularies or detect the labels
        of a classification task.

        Parameters
        ----------
        gold_data: Iterable[Doc]
            The documents to use for initialization.
        exclude: Optional[set]
            The names of components to exclude from initialization.
            This argument will be gradually updated  with the names of initialized
            components
        """
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        exclude.add(repr_id)
        for name, component in self.named_component_children():
            if hasattr(component, "post_init"):
                component.post_init(gold_data, exclude=exclude)

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        """
        Preprocess the document to extract features that will be used by the
        neural network to perform its predictions.

        Parameters
        ----------
        doc: Doc
            Document to preprocess

        Returns
        -------
        Dict[str, Any]
            Dictionary (optionally nested) containing the features extracted from
            the document.
        """
        return {
            name: component.preprocess(doc)
            for name, component in self.named_component_children()
        }

    def collate(self, batch: Dict[str, Any]) -> BatchInput:
        """
        Collate the batch of features into a single batch of tensors that can be
        used by the forward method of the component.

        Parameters
        ----------
        batch: Dict[str, Any]
            Batch of features

        Returns
        -------
        BatchInput
            Dictionary (optionally nested) containing the collated tensors
        """
        return {
            name: component.collate(batch[name])
            for name, component in self.named_component_children()
        }

    def batch_to_device(
        self,
        batch: BatchInput,
        device: Optional[Union[str, torch.device]],
    ) -> BatchInput:
        """
        Move the batch of tensors to the specified device.

        Parameters
        ----------
        batch: BatchInput
            Batch of tensors
        device: Optional[Union[str, torch.device]]
            Device to move the tensors to

        Returns
        -------
        BatchInput
        """
        return {
            name: (
                value.to(device)
                if hasattr(value, "to")
                else getattr(self, name).batch_to_device(value, device=device)
                if hasattr(self, name)
                else value
            )
            for name, value in batch.items()
        }

    def forward(self, batch: BatchInput) -> BatchOutput:
        """
        Perform the forward pass of the neural network.

        Parameters
        ----------
        batch: BatchInput
            Batch of tensors (nested dictionary) computed by the collate method

        Returns
        -------
        BatchOutput
        """
        raise NotImplementedError()

    def module_forward(self, *args, **kwargs):
        """
        This is a wrapper around `torch.nn.Module.__call__` to avoid conflict
        with the [`Component.__call__`][edspdf.component.Component.__call__]
        method.
        """
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def make_batch(
        self,
        docs: Sequence[Doc],
        supervision: bool = False,
    ) -> Dict[str, Sequence[Any]]:
        """
        Convenience method to preprocess a batch of documents and collate them
        Features corresponding to the same path are grouped together in a list,
        under the same key.

        Parameters
        ----------
        docs: Sequence[Doc]
            Batch of documents
        supervision: bool
            Whether to extract supervision features or not

        Returns
        -------
        Dict[str, Sequence[Any]]
        """
        batch = [
            (self.preprocess_supervised(doc) if supervision else self.preprocess(doc))
            for doc in docs
        ]
        return decompress_dict(list(batch_compress_dict(batch)))

    def batch_process(self, docs: Sequence[Doc]) -> Sequence[Doc]:
        """
        Process a batch of documents using the neural network.
        This differs from the `pipe` method in that it does not return an
        iterator, but executes the component on the whole batch at once.

        Parameters
        ----------
        docs: Sequence[Doc]
            Batch of documents

        Returns
        -------
        Sequence[Doc]
            Batch of updated documents
        """
        device = next((p.device for p in self.parameters()), "cpu")
        with torch.no_grad():
            batch = self.make_batch(docs)
            inputs = self.collate(batch)
            inputs = self.batch_to_device(inputs, device=device)
            if hasattr(self, "compiled"):
                res = self.compiled(inputs)
            else:
                res = self.module_forward(inputs)
            docs = self.postprocess(docs, res)
            return docs

    def postprocess(self, docs: Sequence[Doc], batch: BatchOutput) -> Sequence[Doc]:
        """
        Update the documents with the predictions of the neural network.
        By default, this is a no-op.

        Parameters
        ----------
        docs: Sequence[Doc]
            Batch of documents
        batch: BatchOutput
            Batch of predictions, as returned by the forward method

        Returns
        -------
        Sequence[Doc]
        """
        return docs

    # Same as preprocess but with gold supervision data
    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        """
        Preprocess the document to extract features that will be used by the
        neural network to perform its training.
        By default, this returns the same features as the `preprocess` method.

        Parameters
        ----------
        doc: Doc
            Document to preprocess

        Returns
        -------
        Dict[str, Any]
            Dictionary (optionally nested) containing the features extracted from
            the document.
        """
        return self.preprocess(doc)

    def pipe(self, docs: Iterable[Doc], batch_size=1) -> Iterable[Doc]:
        """
        Applies the component on a collection of documents. It is recommended to use
        the [`Pipeline.pipe`][edsnlp.core.pipeline.Pipeline.pipe] method instead of this
        one to apply a pipeline on a collection of documents, to benefit from the
        caching of intermediate results.

        Parameters
        ----------
        docs: Iterable[Doc]
            Input docs
        batch_size: int
            Batch size to use when making batched to be process at once
        """
        for batch in batchify(docs, batch_size=batch_size):
            yield from self.batch_process(batch)

    def __call__(self, doc: Doc) -> Doc:
        """
        Applies the component on a single doc.
        For multiple documents, prefer batch processing via the
        [batch_process][edspdf.trainable_pipe.TrainablePipe.batch_process] method.
        In general, prefer the [Pipeline][edspdf.pipeline.Pipeline] methods

        Parameters
        ----------
        doc: Doc

        Returns
        -------
        Doc
        """
        return self.batch_process([doc])[0]

    def to_disk(self, path, *, exclude: Optional[Set[str]]):
        if object.__repr__(self) in exclude:
            return
        exclude.add(object.__repr__(self))
        overrides = {}
        for name, component in self.named_component_children():
            if hasattr(component, "to_disk"):
                pipe_overrides = component.to_disk(path / name, exclude=exclude)
                if pipe_overrides:
                    overrides[name] = pipe_overrides
        tensor_dict = {
            n: p
            for n, p in (*self.named_parameters(), *self.named_buffers())
            if object.__repr__(p) not in exclude
        }
        os.makedirs(path, exist_ok=True)
        safetensors.torch.save_file(tensor_dict, path / "parameters.safetensors")
        exclude.update(object.__repr__(p) for p in tensor_dict.values())
        return overrides

    def from_disk(self, path, exclude: Optional[Set[str]]):
        if object.__repr__(self) in exclude:
            return
        exclude.add(object.__repr__(self))
        for name, component in self.named_component_children():
            if hasattr(component, "from_disk"):
                component.from_disk(path / name, exclude=exclude)
        tensor_dict = safetensors.torch.load_file(path / "parameters.safetensors")
        self.load_state_dict(tensor_dict, strict=False)
