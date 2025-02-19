from __future__ import annotations

import abc
import random
import sys
import textwrap
import warnings
from collections import namedtuple
from copy import copy
from functools import wraps
from inspect import isgeneratorfunction, signature
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

from confit import VisibleDeprecationWarning
from typing_extensions import Literal

import edsnlp.data
from edsnlp.utils.batching import BatchBy, BatchFn, BatchSizeArg, batchify, batchify_fns
from edsnlp.utils.collections import flatten, flatten_once, shuffle
from edsnlp.utils.stream_sentinels import StreamSentinel

if TYPE_CHECKING:
    import torch

    from edsnlp import Pipeline
    from edsnlp.core.torch_component import TorchComponent
    from edsnlp.data.base import BaseReader, BaseWriter, BatchWriter


def deep_isgeneratorfunction(x):
    if hasattr(x, "__call__"):
        return isgeneratorfunction(x) or isgeneratorfunction(x.__call__)
    elif hasattr(x, "batch_process"):
        return isgeneratorfunction(x.batch_process) or isgeneratorfunction(
            x.batch_process.__call__
        )
    raise ValueError(f"{x} does not have a __call__ or batch_process method.")


CONTEXT = [{}]

T = TypeVar("T")


def with_non_default_args(fn: T) -> T:
    @wraps(fn)
    def wrapper(self, **kwargs):
        return fn(self, **kwargs, _non_default_args=kwargs.keys())

    return wrapper


Batchable = namedtuple("Batchable", ["batch_process"])


def make_kwargs_str(kwargs, first=True):
    pre_sep, join_sep = ("", ", ") if first else (", ", "")
    return join_sep.join(pre_sep + f"{k}={v!r}" for k, v in kwargs.items())


class Op(abc.ABC):
    elementwise: bool

    def __call__(self, items):
        raise NotImplementedError()


class FlattenOp(Op):
    elementwise = False

    def __call__(self, items):
        return flatten(items)

    def __repr__(self):
        return "flatten()"


class UnbatchifyOp(Op):
    elementwise = True

    def __call__(self, items):
        return flatten_once(items)

    def __repr__(self):
        return "unbatchify()"


class BatchifyOp(Op):
    elementwise = True

    def __init__(
        self,
        size,
        batch_fn: BatchFn,
        sentinel_mode: Optional[Literal["drop", "split", "auto"]] = None,
    ):
        if batch_fn is None:
            if size is None:
                size = None
                batch_fn = None
            else:
                batch_fn = batchify_fns["docs"]
        self.size = size
        self.batch_fn = batch_fn
        self.sentinel_mode = sentinel_mode

    def __call__(self, items):
        assert self.sentinel_mode != "auto"
        return self.batch_fn(
            items,
            self.size,
            **{"sentinel_mode": self.sentinel_mode}
            if self.sentinel_mode is not None
            else {},
        )

    def __repr__(self):
        return (
            "batchify("
            f"size={self.size}, "
            f"fn={self.batch_fn}, "
            f"sentinel_mode={self.sentinel_mode})"
        )


class MapOp(Op):
    def __init__(self, pipe, kwargs, context=None):
        self.pipe = pipe
        self.kwargs = kwargs
        self.is_generator = deep_isgeneratorfunction(pipe)
        self.elementwise = not self.is_generator
        self.context = context or {}

    def __call__(self, items):
        for item in items:
            if isinstance(item, StreamSentinel):
                yield item
                continue

            CONTEXT[0], old = self.context, CONTEXT[0]
            res = self.pipe(item, **self.kwargs)
            CONTEXT[0] = old

            if self.is_generator:
                yield from res
            else:
                yield res

    def __repr__(self):
        if hasattr(self.pipe, "__self__"):
            op_str = f"{self.pipe.__name__}[{object.__repr__(self.pipe.__self__)}]"
        else:
            op_str = object.__repr__(self.pipe)
        return "map({}{})".format(op_str, make_kwargs_str(self.kwargs, False))


class MapBatchesOp(Op):
    def __init__(self, pipe, kwargs, context=None, elementwise=False):
        self.pipe = pipe
        self.kwargs = kwargs
        self.is_generator = deep_isgeneratorfunction(pipe)
        if elementwise and self.is_generator:
            raise ValueError("Cannot use elementwise=True with a generator function")
        self.elementwise = elementwise
        self.context = context or {}

    def __call__(self, batches):
        if hasattr(self.pipe, "batch_process"):
            for batch in batches:
                if isinstance(batch, StreamSentinel):
                    yield batch
                    continue
                CONTEXT[0], old = self.context, CONTEXT[0]
                res = self.pipe.batch_process(batch, **self.kwargs)
                CONTEXT[0] = old
                res = list(res) if self.is_generator else (res,)
                yield from res
        else:
            for batch in batches:
                if isinstance(batch, StreamSentinel):
                    yield batch
                    continue
                results = []
                for item in batch:
                    CONTEXT[0], old = self.context, CONTEXT[0]
                    res = (
                        item
                        if isinstance(item, StreamSentinel)
                        else self.pipe(item, **self.kwargs)
                    )
                    CONTEXT[0] = old
                    res = list(res) if self.is_generator else (res,)
                    results.extend(res)
                yield results

    def __repr__(self):
        pipe = (
            self.pipe.batch_process if isinstance(self.pipe, Batchable) else self.pipe
        )
        if hasattr(pipe, "__self__"):
            op_str = f"{pipe.__name__}[{object.__repr__(pipe.__self__)}]"
        else:
            op_str = object.__repr__(pipe)

        return f"map_batches_op({op_str}{make_kwargs_str(self.kwargs, False)})"


class QuickTorchPipe:
    def __init__(self, prepare_batch, forward, postprocess, elementwise=False):
        self.prepare_batch = prepare_batch
        self.forward = forward
        self.postprocess = postprocess
        self.elementwise = elementwise

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def batch_process(self, batch):
        res = self.forward(self.prepare_batch(batch, None))
        return self.postprocess(batch, res) if self.postprocess is not None else res

    def enable_cache(self, cache_id=None):
        pass

    def disable_cache(self, cache_id=None):
        pass


class Stage:
    def __init__(self, cpu_ops: List[Op], gpu_op: Optional[TorchComponent]):
        self.cpu_ops = cpu_ops
        self.gpu_op = gpu_op

    def __repr__(self):
        args_str = ",\n".join(textwrap.indent(repr(op), "    ") for op in self.cpu_ops)
        return (
            f"Stage(\n"
            f"  cpu_ops=[\n{args_str}\n  ],\n"
            f"  gpu_op={object.__repr__(self.gpu_op) if self.gpu_op else None})"
        )


class MetaStream(type):
    def __getattr__(self, item):
        if item in edsnlp.data.__all__:
            fn = getattr(edsnlp.data, item)
            setattr(self, item, fn)
            return fn
        raise AttributeError(item)

    def __dir__(self):  # pragma: no cover
        return (*super().__dir__(), *edsnlp.data.__all__)


class Stream(metaclass=MetaStream):
    def __init__(
        self,
        reader: Optional[BaseReader] = None,
        writer: Optional[Union[BaseWriter, BatchWriter]] = None,
        ops: List[Any] = [],
        config: Optional[Dict] = None,
    ):
        self.reader = reader
        self.writer = writer
        self.ops: List[Op] = ops
        self.config = config or {}

    @classmethod
    def validate_batching(cls, batch_size, batch_by):
        if isinstance(batch_size, str):
            if batch_by is not None:
                raise ValueError(
                    "Cannot use both a batch_size expression and a batch_by function"
                )
            batch_size, batch_by = BatchSizeArg.validate(batch_size)
        if batch_size is not None and not isinstance(batch_size, int):
            raise ValueError(
                f"Invalid batch_size (must be an integer or None): {batch_size}"
            )
        if (
            batch_by is not None
            and batch_by not in batchify_fns
            and not callable(batch_by)
        ):
            raise ValueError(f"Invalid batch_by function: {batch_by}")
        return batch_size, batch_by

    @property
    def batch_size(self):
        return self.config.get("batch_size", None)

    @property
    def batch_by(self):
        return self.config.get("batch_by", None)

    @property
    def disable_implicit_parallelism(self):
        return self.config.get("disable_implicit_parallelism", True)

    @property
    def num_cpu_workers(self):
        return self.config.get("num_cpu_workers")

    @property
    def num_gpu_workers(self):
        return self.config.get("num_gpu_workers")

    @property
    def gpu_worker_devices(self):
        return self.config.get("gpu_worker_devices")

    @property
    def cpu_worker_devices(self):
        return self.config.get("cpu_worker_devices")

    @property
    def autocast(self):
        return self.config.get("autocast", True)

    @property
    def backend(self):
        backend = self.config.get("backend")
        return {"mp": "multiprocessing"}.get(backend, backend)

    @property
    def show_progress(self):
        return self.config.get("show_progress", False)

    @property
    def process_start_method(self):
        return self.config.get("process_start_method")

    @property
    def deterministic(self):
        return self.config.get("deterministic", True)

    # noinspection PyIncorrectDocstring
    @with_non_default_args
    def set_processing(
        self,
        batch_size: Optional[Union[int, str]] = None,
        batch_by: BatchBy = None,
        split_into_batches_after: str = None,
        num_cpu_workers: Optional[int] = None,
        num_gpu_workers: Optional[int] = None,
        disable_implicit_parallelism: bool = True,
        backend: Optional[Literal["simple", "multiprocessing", "mp", "spark"]] = None,
        autocast: Union[bool, Any] = None,
        show_progress: bool = False,
        gpu_pipe_names: Optional[List[str]] = None,
        process_start_method: Optional[Literal["fork", "spawn"]] = None,
        gpu_worker_devices: Optional[List[str]] = None,
        cpu_worker_devices: Optional[List[str]] = None,
        deterministic: bool = True,
        chunk_size: int = None,
        sort_chunks: bool = False,
        _non_default_args: Iterable[str] = (),
    ) -> "Stream":
        """
        Parameters
        ----------
        batch_size: Optional[Union[int, str]]
            The batch size. Can also be a batching expression like
            "32 docs", "1024 words", "dataset", "fragment", etc.
        batch_by: BatchBy
            Function to compute the batches. If set, it should take an iterable of
            documents and return an iterable of batches. You can also set it to
            "docs", "words" or "padded_words" to use predefined batching functions.
            Defaults to "docs".
        num_cpu_workers: int
            Number of CPU workers. A CPU worker handles the non deep-learning components
            and the preprocessing, collating and postprocessing of deep-learning
            components. If no GPU workers are used, the CPU workers also handle the
            forward call of the deep-learning components.
        num_gpu_workers: Optional[int]
            Number of GPU workers. A GPU worker handles the forward call of the
            deep-learning components. Only used with "multiprocessing" backend.
        disable_implicit_parallelism: bool
            Whether to disable OpenMP and Huggingface tokenizers implicit parallelism in
            multiprocessing mode. Defaults to True.
        backend: Optional[Literal["simple", "multiprocessing", "spark"]]
            The backend to use for parallel processing. If not set, the backend is
            automatically selected based on the input data and the number of workers.

            - "simple" is the default backend and is used when `num_cpu_workers` is 1
                and `num_gpu_workers` is 0.
            - "multiprocessing" is used when `num_cpu_workers` is greater than 1 or
                `num_gpu_workers` is greater than 0.
            - "spark" is used when the input data is a Spark dataframe and the output
                writer is a Spark writer.
        autocast: Union[bool, Any]
            Whether to use
            [automatic mixed precision (AMP)](https://pytorch.org/docs/stable/amp.html)
            for the forward pass of the deep-learning components. If True (by default),
            AMP will be used with the default settings. If False, AMP will not be used.
            If a dtype is provided, it will be passed to the `torch.autocast` context
            manager.
        show_progress: Optional[bool]
            Whether to show progress bars (only applicable with "simple" and
            "multiprocessing" backends).
        gpu_pipe_names: Optional[List[str]]
            List of pipe names to accelerate on a GPUWorker, defaults to all pipes
            that inherit from TorchComponent. Only used with "multiprocessing" backend.
            Inferred from the pipeline if not set.
        process_start_method: Optional[Literal["fork", "spawn"]]
            Whether to use "fork" or "spawn" as the start method for the multiprocessing
            backend. The default is "fork" on Unix systems and "spawn" on Windows.

            - "fork" is the default start method on Unix systems and is the fastest
                start method, but it is not available on Windows, can cause issues
                with CUDA and is not safe when using multiple threads.
            - "spawn" is the default start method on Windows and is the safest start
                method, but it is not available on Unix systems and is slower than
                "fork".
        gpu_worker_devices: Optional[List[str]]
            List of GPU devices to use for the GPU workers. Defaults to all available
            devices, one worker per device. Only used with "multiprocessing" backend.
        cpu_worker_devices: Optional[List[str]]
            List of GPU devices to use for the CPU workers. Used for debugging purposes.
        deterministic: bool
            Whether to try and preserve the order of the documents in "multiprocessing"
            mode. If set to `False`, workers will process documents whenever they are
            available in a dynamic fashion, which may result in out-of-order but usually
            faster processing. If set to true, tasks will be distributed in a
            static, round-robin fashion to workers. Defaults to `True`.

        Returns
        -------
        Stream
        """
        kwargs = {k: v for k, v in locals().items() if k in _non_default_args}
        if (
            kwargs.pop("chunk_size", None) is not None
            or kwargs.pop("sort_chunks", None) is not None
        ):
            warnings.warn(
                "chunk_size and sort_chunks are deprecated, use "
                "map_batched(sort_fn, batch_size=chunk_size) instead.",
                VisibleDeprecationWarning,
            )
        if kwargs.pop("split_into_batches_after", None) is not None:
            warnings.warn(
                "split_into_batches_after is deprecated.", VisibleDeprecationWarning
            )
        return Stream(
            reader=self.reader,
            writer=self.writer,
            ops=self.ops,
            config={
                **self.config,
                **{k: v for k, v in kwargs.items() if v is not None},
            },
        )

    @classmethod
    def ensure_stream(cls, data):
        from edsnlp.data.base import IterableReader

        if isinstance(data, cls):
            return data
        return cls(reader=IterableReader(data))

    # For backwards compatibility
    ensure_lazy = ensure_stream

    def map(self, pipe, name: Optional[str] = None, kwargs={}) -> "Stream":
        """
        Maps a callable to the documents. It takes a callable as input and an optional
        dictionary of keyword arguments. The function will be applied to each element
        of the collection. If the callable is a generator function, each element will
        be yielded to the stream as is.

        Parameters
        ----------
        pipe: Any
            The callable to map to the documents.
        kwargs: Dict
            The keyword arguments to pass to the callable.

        Returns
        -------
        Stream
        """
        return Stream(
            reader=self.reader,
            writer=self.writer,
            ops=[*self.ops, MapOp(pipe, kwargs)],
            config=self.config,
        )

    def flatten(self) -> "Stream":
        """
        Flattens the stream.

        Returns
        -------
        Stream
        """
        return Stream(
            reader=self.reader,
            writer=self.writer,
            ops=[*self.ops, FlattenOp()],
            config=self.config,
        )

    def map_batches(
        self,
        pipe,
        name: Optional[str] = None,
        kwargs={},
        batch_size: Optional[Union[int, str]] = None,
        batch_by: BatchBy = None,
    ) -> "Stream":
        """
        Maps a callable to a batch of documents. The callable should take a list of
        inputs. The output of the callable will be flattened if it is a list or
        a generator, or yielded to the stream as is if it is a single output (tuple
        or any other type).

        Parameters
        ----------
        pipe: Any
            The callable to map to the documents.
        kwargs: Dict
            The keyword arguments to pass to the callable.
        batch_size: Optional[Union[int, str]]
            The batch size. Can also be a batching expression like
            "32 docs", "1024 words", "dataset", "fragment", etc.
        batch_by: BatchBy
            Function to compute the batches. If set, it should take an iterable of
            documents and return an iterable of batches. You can also set it to
            "docs", "words" or "padded_words" to use predefined batching functions.
            Defaults to "docs".

        Returns
        -------
        Stream
        """
        batch_size, batch_by = self.validate_batching(batch_size, batch_by)
        batch_fn = batchify_fns.get(batch_by, batch_by)
        infer_batch = batch_size is None and batch_by is None
        ops = list(self.ops)
        if infer_batch and len(ops) and isinstance(ops[-1], UnbatchifyOp):
            ops.pop()
        else:
            ops.append(BatchifyOp(batch_size, batch_fn))
        ops.append(MapBatchesOp(Batchable(pipe), kwargs))
        ops.append(UnbatchifyOp())
        stream = Stream(
            reader=self.reader,
            writer=self.writer,
            ops=ops,
            config=self.config,
        )
        stream.validate_ops(ops=stream.ops, update=False)
        return stream

    def batchify(
        self,
        batch_size: Optional[Union[int, str]] = None,
        batch_by: BatchBy = None,
    ) -> "Stream":
        """
        Accumulates the documents into batches and yield each batch to the stream.

        Parameters
        ----------
        batch_size: Optional[Union[int, str]]
            The batch size. Can also be a batching expression like
            "32 docs", "1024 words", "dataset", "fragment", etc.
        batch_by: BatchBy
            Function to compute the batches. If set, it should take an iterable of
            documents and return an iterable of batches. You can also set it to
            "docs", "words" or "padded_words" to use predefined batching functions.
            Defaults to "docs".

        Returns
        -------
        Stream
        """
        batch_size, batch_by = self.validate_batching(batch_size, batch_by)
        batch_fn = batchify_fns.get(batch_by, batch_by)
        ops = list(self.ops)
        ops.append(BatchifyOp(batch_size, batch_fn))
        stream = Stream(
            reader=self.reader,
            writer=self.writer,
            ops=ops,
            config=self.config,
        )
        stream.validate_ops(ops=stream.ops, update=False)
        return stream

    def map_gpu(
        self,
        prepare_batch: Callable[[List, Union[str, torch.device]], Any],
        forward: Callable[[Any], Any],
        postprocess: Optional[Callable[[List, Any], Any]] = None,
        name: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = None,
        batch_by: BatchBy = None,
    ) -> "Stream":
        """
        Maps a deep learning operation to a batch of documents, on a GPU worker.

        Parameters
        ----------
        prepare_batch: Callable[[List, Union[str, torch.device]], Any]
            A callable that takes a list of documents and a device and returns a batch
            of tensors (or anything that can be passed to the `forward` callable). This
            will be called on a CPU-bound worker, and may be parallelized.
        forward: Callable[[Any], Any]
            A callable that takes the output of `prepare_batch` and returns the output
            of the deep learning operation. This will be called on a GPU-bound worker.
        postprocess: Optional[Callable[[List, Any], Any]]
            An optional callable that takes the list of documents and the output of the
            deep learning operation, and returns the final output. This will be called
            on the same CPU-bound worker that called the `prepare_batch` function.
        batch_size: Optional[Union[int, str]]
            The batch size. Can also be a batching expression like
            "32 docs", "1024 words", "dataset", "fragment", etc.
        batch_by: BatchBy
            Function to compute the batches. If set, it should take an iterable of
            documents and return an iterable of batches. You can also set it to
            "docs", "words" or "padded_words" to use predefined batching functions.
            Defaults to "docs".

        Returns
        -------
        Stream
        """
        batch_size, batch_by = self.validate_batching(batch_size, batch_by)
        batch_fn = batchify_fns.get(batch_by, batch_by)
        infer_batch = batch_size is None and batch_by is None
        ops = list(self.ops)
        if infer_batch and len(ops) and isinstance(ops[-1], UnbatchifyOp):
            ops.pop()
        else:
            ops.append(BatchifyOp(batch_size, batch_fn))
        pipe = QuickTorchPipe(prepare_batch, forward, postprocess)
        ops.append(MapBatchesOp(pipe, {}, elementwise=True))
        ops.append(UnbatchifyOp())
        stream = Stream(
            reader=self.reader,
            writer=self.writer,
            ops=ops,
            config=self.config,
        )
        stream.validate_ops(ops=stream.ops, update=False)
        return stream

    def map_pipeline(
        self,
        model: Pipeline,
        batch_size: Optional[Union[int, str]] = None,
        batch_by: BatchBy = None,
    ) -> "Stream":
        """
        Maps a pipeline to the documents, i.e. adds each component of the pipeline to
        the stream operations. This function is called under the hood by `nlp.pipe()`

        Parameters
        ----------
        model: Pipeline
            The pipeline to map to the documents.
        batch_size: Optional[Union[int, str]]
            The batch size. Can also be a batching expression like
            "32 docs", "1024 words", "dataset", "fragment", etc.
        batch_by: BatchBy
            Function to compute the batches. If set, it should take an iterable of
            documents and return an iterable of batches. You can also set it to
            "docs", "words" or "padded_words" to use predefined batching functions.
            Defaults to "docs".

        Returns
        -------
        Stream
        """
        new_ops = []
        tokenizer = model.tokenizer
        for op in self.ops:
            # check if the pipe has a "tokenizer" kwarg and update the kwargs if needed
            op = copy(op)
            if (
                (
                    isinstance(op, MapOp)
                    and "tokenizer" in signature(op.pipe).parameters
                    and "tokenizer" not in op.kwargs
                )
                or (
                    isinstance(op, MapBatchesOp)
                    and hasattr(op.pipe, "batch_process")
                    and "tokenizer" in signature(op.pipe.batch_process).parameters
                    and "tokenizer" not in op.kwargs
                )
                or (
                    isinstance(op, MapBatchesOp)
                    and callable(op.pipe)
                    and "tokenizer" in signature(op.pipe).parameters
                    and "tokenizer" not in op.kwargs
                )
            ):
                op.kwargs["tokenizer"] = tokenizer
            if isinstance(op, (MapOp, MapBatchesOp)):
                op.context["tokenizer"] = tokenizer
            new_ops.append(op)
        new_ops.append(MapOp(model._ensure_doc, {}))
        batch_size, batch_by = self.validate_batching(batch_size, batch_by)
        batch_by = batchify_fns.get(batch_by, batch_by)
        new_ops.append(BatchifyOp(batch_size, batch_by))
        for name, pipe in model.pipeline:
            if name not in model._disabled:
                op = MapBatchesOp(
                    pipe, {}, elementwise=not deep_isgeneratorfunction(pipe)
                )
                new_ops.append(op)
        new_ops.append(UnbatchifyOp())
        config = (
            {**self.config, "batch_size": model.batch_size}
            if self.batch_size is None
            else self.config
        )
        stream = Stream(
            reader=self.reader,
            writer=self.writer,
            ops=new_ops,
            config=config,
        )
        stream.validate_ops(ops=stream.ops, update=False)
        return stream

    def shuffle(
        self,
        batch_size: Optional[Union[str, int]] = None,
        batch_by: Optional[str, BatchFn] = None,
        seed: Optional[int] = None,
        shuffle_reader: Optional[Union[bool, str]] = None,
    ) -> "Stream":
        """
        Shuffles the stream by accumulating the documents into batches and shuffling
        the batches. We try to optimize and avoid the accumulation by shuffling items
        directly in the reader, but if some upstream operations are not elementwise
        or if the reader is not compatible with the batching mode, we have to accumulate
        the documents into batches and shuffle the batches.

        For instance, imagine a reading from list of 2 very large documents and applying
        an operation to split the documents into sentences. Shuffling only in the
        reader, then applying the split operation would not shuffle the sentences across
        documents and may lead to a lack of randomness when training a model. Think of
        this as having lumps after mixing your data. In our case, we detect that the
        split op is not elementwise and trigger the accumulation of sentences into
        batches after their generation before shuffling the batches.

        Parameters
        ----------
        batch_size: Optional[Union[int, str]]
            The batch size. Can also be a batching expression like
            "32 docs", "1024 words", "dataset", "fragment", etc.
        batch_by: BatchBy
            Function to compute the batches. If set, it should take an iterable of
            documents and return an iterable of batches. You can also set it to
            "docs", "words" or "padded_words" to use predefined batching functions.
            Defaults to "docs".
        seed: Optional[int]
            The seed to use for shuffling.
        shuffle_reader: Optional[bool]
            Whether to shuffle the reader. Defaults to True if the reader is compatible
            with the batch_by mode, False otherwise.

        Returns
        -------
        Stream
        """
        batch_size, batch_by = self.validate_batching(batch_size, batch_by)
        if batch_by is None and batch_size is None:
            batch_by = "dataset"
        if shuffle_reader is None or shuffle_reader is True:
            possible_shuffle_reader = (
                batch_by
                if batch_by in self.reader.emitted_sentinels and not self.reader.shuffle
                else False
            )
            if not possible_shuffle_reader and shuffle_reader:
                # Maybe should we be more explicit about why we cannot shuffle ?
                raise ValueError(
                    "You cannot shuffle the reader given the current stream and the "
                    f"batching mode {batch_by!r}."
                )
            shuffle_reader = possible_shuffle_reader
        stream = self
        if shuffle_reader:
            if shuffle_reader not in self.reader.emitted_sentinels:
                raise ValueError(f"Cannot shuffle by {shuffle_reader}")
            stream = Stream(
                reader=copy(stream.reader),
                writer=stream.writer,
                ops=stream.ops,
                config=stream.config,
            )
            stream.reader.shuffle = shuffle_reader
            # Ensure that we have a "deterministic" random seed, meaning
            # if the user sets a global seed before in the program and execute the
            # same program twice, the shuffling should be the same in both cases.
            # This is not garanteed by just creating random.Random() which does not
            # account for the global seed.
            if seed is not None:
                stream.reader.rng = random.Random(seed)
            # Else, if seed is None, then the reader rng stays the same
        if any(not op.elementwise for op in self.ops) or shuffle_reader != batch_by:
            stream = stream.map_batches(
                pipe=shuffle,
                batch_size=batch_size,
                batch_by=batch_by,
                kwargs={"rng": random.Random(seed)},
            )
        stream.validate_ops(ops=stream.ops, update=False)
        return stream

    def loop(self) -> "Stream":
        """
        Loops over the stream indefinitely.

        Note that we cycle over items produced by the reader, not the items produced by
        the stream operations. This means that the stream operations will be applied to
        the same items multiple times, and may produce different results if they are
        non-deterministic. This also mean that calling this function will have the same
        effect regardless of the operations applied to the stream before calling it, ie:

        ```
        stream.loop().map(...)
        # is equivalent to
        stream.map(...).loop()
        ```

        Returns
        -------
        Stream
        """
        stream = Stream(
            reader=copy(self.reader),
            writer=self.writer,
            ops=self.ops,
            config=self.config,
        )
        stream.reader.loop = True
        return stream

    def write(self, writer: BaseWriter, execute: bool = True) -> Any:
        if self.writer is not None:
            raise ValueError("A writer is already set.")
        stream = Stream(
            reader=self.reader,
            writer=writer,
            ops=self.ops,
            config=self.config,
        )
        return stream.execute() if execute else stream

    def execute(self):
        import edsnlp.processing

        backend = self.backend
        if backend is None:
            try:
                SparkReader = sys.modules.get("edsnlp.data.spark").SparkReader
                SparkWriter = sys.modules.get("edsnlp.data.spark").SparkWriter
            except (KeyError, AttributeError):  # pragma: no cover
                SparkReader = SparkWriter = None
            if (
                SparkReader
                and isinstance(self.reader, SparkReader)
                and SparkWriter
                and (self.writer is None or isinstance(self.writer, SparkWriter))
            ):
                backend = "spark"
            elif (
                self.num_cpu_workers is not None or self.num_gpu_workers is not None
            ) and (
                self.num_cpu_workers is not None
                and self.num_cpu_workers > 0
                or self.num_gpu_workers is not None
                and self.num_gpu_workers > 0
            ):
                backend = "multiprocessing"
            else:
                backend = "simple"
        execute = getattr(edsnlp.processing, f"execute_{backend}_backend")
        return execute(self)

    def __iter__(self):
        return iter(self.execute())

    def torch_components(self) -> Iterable["TorchComponent"]:
        """
        Yields components that are PyTorch modules.

        Returns
        -------
        Iterable['edsnlp.core.torch_component.TorchComponent']
        """
        for op in self.ops:
            if hasattr(op, "pipe") and hasattr(op.pipe, "forward"):
                yield op.pipe

    def train(self, mode=True):
        """
        Enables training mode on pytorch modules

        Parameters
        ----------
        mode: bool
            Whether to enable training or not
        """

        class context:
            def __enter__(self):
                pass

            def __exit__(ctx_self, type, value, traceback):
                for proc in procs:
                    proc.train(was_training[proc])

        procs = [x for x in self.torch_components() if hasattr(x, "train")]
        was_training = {proc: proc.training for proc in procs}
        for proc in procs:
            proc.train(mode)

        return context()

    def eval(self):
        """
        Enables evaluation mode on pytorch modules
        """
        return self.train(False)

    def worker_copy(self):
        return Stream(
            reader=self.reader.worker_copy(),
            writer=self.writer,
            ops=self.ops,
            config=self.config,
        )

    def __dir__(self):  # pragma: no cover
        return (*super().__dir__(), *edsnlp.data.__all__)

    def __getattr__(self, item):
        return getattr(Stream, item).__get__(self)

    def _make_stages(self, split_torch_pipes: bool) -> List[Stage]:
        current_ops = []
        stages = []

        ops = [copy(op) for op in self.ops]

        for op in ops:
            if (
                isinstance(op, MapBatchesOp)
                and hasattr(op.pipe, "forward")
                and split_torch_pipes
            ):
                stages.append(Stage(current_ops, op.pipe))
                current_ops = []
            else:
                current_ops.append(op)

        if len(current_ops) or len(stages) == 0:
            stages.append(Stage(current_ops, None))

        self.validate_ops(ops=ops, update=True)

        return stages

    def validate_ops(self, ops, update: bool = False):
        # Check batchify requirements
        requires_sentinels = set()

        self_batch_size, self_batch_by = self.validate_batching(
            self.batch_size, self.batch_by
        )
        if self_batch_by is None:
            self_batch_by = "docs"
        if self_batch_size is None:
            self_batch_size = 1
        self_batch_fn = batchify_fns.get(self_batch_by, self_batch_by)

        if hasattr(self.writer, "batch_fn") and hasattr(
            self.writer.batch_fn, "requires_sentinel"
        ):
            requires_sentinels.add(self.writer.batch_fn.requires_sentinel)

        for op in reversed(ops):
            if isinstance(op, BatchifyOp):
                if op.batch_fn is None and op.size is None:
                    batch_size = self_batch_size
                    batch_fn = self_batch_fn
                elif op.batch_fn is None:
                    batch_size = op.size
                    batch_fn = batchify
                else:
                    batch_size = op.size
                    batch_fn = op.batch_fn
                sentinel_mode = op.sentinel_mode or (
                    "auto"
                    if "sentinel_mode" in signature(batch_fn).parameters
                    else None
                )
                if sentinel_mode == "auto":
                    sentinel_mode = "split" if requires_sentinels else "drop"
                if requires_sentinels and sentinel_mode == "drop":
                    raise ValueError(
                        f"Operation {op} drops the stream sentinel values "
                        f"(markers for the end of a dataset or a dataset "
                        f"fragment), but some downstream operation(s) require "
                        f"the following sentinel values: {requires_sentinels}. "
                        f"Ensure that you do not set `sentinel_mode='drop'` on "
                        f"any upstream batching operation."
                    )
                if update:
                    op.size = batch_size
                    op.batch_fn = batch_fn
                    op.sentinel_mode = sentinel_mode

                if hasattr(op.batch_fn, "requires_sentinel"):
                    requires_sentinels.add(op.batch_fn.requires_sentinel)

        sentinel_str = ", ".join(requires_sentinels)
        if requires_sentinels and self.backend == "spark":
            raise ValueError(
                f"Some operations require sentinel values ({sentinel_str}), "
                f"but the Spark backend does not support sentinel values."
            )
        if requires_sentinels and not self.deterministic:
            raise ValueError(
                f"Some operations require sentinel values ({sentinel_str}), "
                f"but these are not supported in when `deterministic=False`."
            )
        if not (requires_sentinels <= self.reader.emitted_sentinels):
            raise ValueError(
                f"Some operations require sentinel values ({sentinel_str}), "
                f"but the reader does not emit these values "
                f"({', '.join(self.reader.emitted_sentinels)})."
            )

    def __repr__(self):
        ops_str = ",\n".join(textwrap.indent(repr(op), "    ") for op in self.ops)
        if ops_str:
            ops_str = "\n" + ops_str + "\n  "
        return (
            f"Stream(\n"
            f"  reader={self.reader},\n"
            f"  ops=[{ops_str}],\n"
            f"  writer={self.writer})\n"
        )

    if TYPE_CHECKING:
        from edsnlp.data import to_iterable as to_iterable  # noqa: F401
        from edsnlp.data import to_pandas as to_pandas  # noqa: F401
        from edsnlp.data import to_polars as to_polars  # noqa: F401
        from edsnlp.data import to_spark as to_spark  # noqa: F401
        from edsnlp.data import write_brat as write_brat  # noqa: F401
        from edsnlp.data import write_json as write_json  # noqa: F401
        from edsnlp.data import write_parquet as write_parquet  # noqa: F401
        from edsnlp.data import write_standoff as write_standoff  # noqa: F401


# For backwards compatibility
LazyCollection = Stream
