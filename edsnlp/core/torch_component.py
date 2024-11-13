import copyreg
import io
import os
import pickle
from enum import Enum
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import safetensors.torch
import spacy
import torch
import xxhash
from spacy.tokens import Doc

from edsnlp.pipes.base import BaseComponent, BaseComponentMeta
from edsnlp.utils.collections import batch_compress_dict, batchify, decompress_dict

BatchInput = TypeVar("BatchInput", bound=Dict[str, Any])
BatchOutput = TypeVar("BatchOutput", bound=Dict[str, Any])
Scorer = Callable[[Sequence[Tuple[Doc, Doc]]], Union[float, Dict[str, Any]]]

ALL_CACHES = object()
_caches = {}
dispatch_table = copyreg.dispatch_table.copy()


def simple_hash_pickle(x):
    return (hash, (hash(x),))


dispatch_table[spacy.tokens.Doc] = simple_hash_pickle
dispatch_table[spacy.tokens.Span] = simple_hash_pickle
dispatch_table[spacy.tokens.Token] = simple_hash_pickle


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


def hash_inputs(inputs):
    res = io.BytesIO()
    p = pickle.Pickler(res)
    p.dispatch_table = dispatch_table
    hashed = xxhash.xxh3_64()
    p.dump(inputs)
    hashed.update(res.getvalue())
    return hashed.hexdigest()


def cached(key, store_key=False):
    def wrapper(fn):
        @wraps(fn)
        def wrapped(self: "TorchComponent", *args, **kwargs):
            if self._current_cache_id is None or len(args) == 0:
                return fn(self, *args, **kwargs)
            cache_key = (
                fn.__name__,
                f"{self.__class__.__name__}<{id(self)}>",
                key(self, *args, **kwargs),
            )
            cache = _caches[self._current_cache_id]
            if cache_key in cache:
                return cache[cache_key]
            res = fn(self, *args, **kwargs)
            cache[cache_key] = res
            if store_key:
                res["__cache_key__"] = cache_key
            return res

        wrapped._cached = fn
        return wrapped

    return wrapper


def _cached_preprocess(fn):
    if hasattr(fn, "_cached"):  # pragma: no cover
        return fn
    return cached(
        lambda self, *args, **kwargs: hash_inputs(
            (*args, sorted(kwargs.items(), key=lambda x: x[0]))
        )
    )(fn)


def _cached_preprocess_supervised(fn):
    if hasattr(fn, "_cached"):  # pragma: no cover
        return fn
    return cached(
        lambda self, *args, **kwargs: hash_inputs(
            (*args, sorted(kwargs.items(), key=lambda x: x[0]))
        )
    )(fn)


def _cached_collate(fn):
    if hasattr(fn, "_cached"):  # pragma: no cover
        return fn
    return cached(
        lambda self, batch, *args, **kwargs: hash_batch(batch), store_key=True
    )(fn)


def _cached_forward(fn):
    if hasattr(fn, "_cached"):  # pragma: no cover
        return fn
    return cached(lambda self, batch, *args, **kwargs: hash_batch(batch))(fn)


def _cached_batch_to_device(fn):
    return cached(
        lambda self, batch, device: (
            hash_batch(batch),
            device,
        )
    )(fn)


class TorchComponentMeta(BaseComponentMeta):
    def __new__(mcs, name, bases, class_dict):
        if "preprocess" in class_dict:
            class_dict["preprocess"] = _cached_preprocess(class_dict["preprocess"])
        if "preprocess_supervised" in class_dict:
            class_dict["preprocess_supervised"] = _cached_preprocess_supervised(
                class_dict["preprocess_supervised"]
            )
        if "collate" in class_dict:
            class_dict["collate"] = _cached_collate(class_dict["collate"])
        if "batch_to_device" in class_dict:
            class_dict["batch_to_device"] = _cached_batch_to_device(
                class_dict["batch_to_device"]
            )
        if "forward" in class_dict:
            class_dict["forward"] = _cached_forward(class_dict["forward"])

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

    def preprocess(self, doc: Doc, **kwargs) -> Dict[str, Any]:
        """
        Preprocess the document to extract features that will be used by the
        neural network and its subcomponents on to perform its predictions.

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
            name: component.preprocess(doc, **kwargs)
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

        def rec(x):
            if hasattr(x, "to"):
                return x.to(device)
            if isinstance(x, dict):
                return {name: rec(value) for name, value in x.items()}
            if isinstance(x, (list, tuple, set)):
                return type(x)(rec(value) for value in x)
            return x

        return rec(batch)

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

    def module_forward(self, *args, **kwargs):  # pragma: no cover
        """
        This is a wrapper around `torch.nn.Module.__call__` to avoid conflict
        with the components `__call__` method.
        """
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def prepare_batch(
        self,
        docs: Sequence[Doc],
        supervision: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> BatchInput:
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
        device: Optional[Union[str, torch.device]]
            Device to move the tensors to

        Returns
        -------
        Dict[str, Sequence[Any]]
        """
        batch = [
            (self.preprocess_supervised(doc) if supervision else self.preprocess(doc))
            for doc in docs
        ]
        batch = decompress_dict(list(batch_compress_dict(batch)))
        batch = self.collate(batch)
        batch = self.batch_to_device(batch, device=device)
        return batch

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
        with torch.no_grad():
            inputs = [self.preprocess(doc) for doc in docs]
            batch = decompress_dict(list(batch_compress_dict(inputs)))
            batch = self.collate(batch)
            batch = self.batch_to_device(batch, device=self.device)
            if hasattr(self, "compiled"):
                res = self.compiled(batch)
            else:
                res = self(batch)
            docs = self.postprocess(docs, res, inputs)
            return docs

    def postprocess(
        self, docs: Sequence[Doc], results: BatchOutput, inputs: List[Dict[str, Any]]
    ) -> Sequence[Doc]:
        """
        Update the documents with the predictions of the neural network.
        By default, this is a no-op.

        Parameters
        ----------
        docs: Sequence[Doc]
            List of documents to update
        results: BatchOutput
            Batch of predictions, as returned by the forward method
        inputs: BatchInput
            List of preprocessed features, as returned by the preprocess method

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

    if TYPE_CHECKING:

        def __call__(self, batch: BatchInput) -> BatchOutput: ...

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
