from __future__ import annotations

import abc
import inspect
import sys
import textwrap
import types
import warnings
from collections import namedtuple
from copy import copy
from functools import wraps
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
from edsnlp.utils.batching import batchify_fns
from edsnlp.utils.collections import flatten, flatten_once

if TYPE_CHECKING:
    import torch

    from edsnlp import Pipeline
    from edsnlp.core.torch_component import TorchComponent
    from edsnlp.data.base import BaseReader, BaseWriter


class _InferType:
    # Singleton is important since the INFER object may be passed to
    # other processes, i.e. pickled, depickled, while it should
    # always be the same object.
    instance = None

    def __repr__(self):
        return "INFER"

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __bool__(self):
        return False


INFER = _InferType()

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
    def __call__(self, items):
        raise NotImplementedError()


class FlattenOp(Op):
    def __call__(self, items):
        return flatten(items)

    def __repr__(self):
        return "flatten()"


class UnbatchifyOp(Op):
    def __call__(self, items):
        return flatten_once(items)

    def __repr__(self):
        return "unbatchify()"


class BatchifyOp(Op):
    def __init__(self, size, batch_fn: Callable[[Iterable, int], Iterable]):
        if batch_fn is None:
            if size is None:
                size = INFER
                batch_fn = INFER
            else:
                batch_fn = batchify_fns["docs"]
        self.size = size
        self.batch_fn = batch_fn

    def __call__(self, items):
        return self.batch_fn(items, self.size)

    def __repr__(self):
        return "batchify({}, {})".format(self.size, self.batch_fn)


class MapOp(Op):
    def __init__(self, pipe, kwargs):
        self.pipe = pipe
        self.kwargs = kwargs

    def __call__(self, items):
        for item in items:
            res = self.pipe(item, **self.kwargs)
            if isinstance(res, types.GeneratorType):
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
    def __init__(self, pipe, kwargs):
        self.pipe = pipe
        self.kwargs = kwargs

    def __call__(self, batches):
        if hasattr(self.pipe, "batch_process"):
            for batch in batches:
                res = self.pipe.batch_process(batch, **self.kwargs)
                res = list(res) if isinstance(res, types.GeneratorType) else (res,)
                yield from res
        else:
            for batch in batches:
                results = []
                for item in batch:
                    res = self.pipe(item, **self.kwargs)
                    res = list(res) if isinstance(res, types.GeneratorType) else (res,)
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


class GPUOp:
    def __init__(self, prepare_batch, forward, postprocess):
        self.prepare_batch = prepare_batch
        self.forward = forward
        self.postprocess = postprocess

    def __call__(self, batches):
        for batch in batches:
            res = self.forward(self.prepare_batch(batch, None))
            yield self.postprocess(batch, res) if self.postprocess is not None else res

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
        writer: Optional[BaseWriter] = None,
        ops: List[Any] = [],
        config: Dict = {},
    ):
        self.reader = reader
        self.writer = writer
        self.ops: List[Op] = ops
        self.config = config

    @property
    def batch_size(self):
        return self.config.get("batch_size", 1)

    @property
    def batch_by(self):
        return self.config.get("batch_by", "docs")

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
        batch_size: int = INFER,
        batch_by: Literal["docs", "words", "padded_words"] = "docs",
        split_into_batches_after: str = INFER,
        num_cpu_workers: Optional[int] = INFER,
        num_gpu_workers: Optional[int] = INFER,
        disable_implicit_parallelism: bool = True,
        backend: Optional[Literal["simple", "multiprocessing", "mp", "spark"]] = INFER,
        autocast: Union[bool, Any] = INFER,
        show_progress: bool = False,
        gpu_pipe_names: Optional[List[str]] = INFER,
        process_start_method: Optional[Literal["fork", "spawn"]] = INFER,
        gpu_worker_devices: Optional[List[str]] = INFER,
        cpu_worker_devices: Optional[List[str]] = INFER,
        deterministic: bool = True,
        work_unit: Literal["record", "fragment"] = "record",
        chunk_size: int = INFER,
        sort_chunks: bool = False,
        _non_default_args: Iterable[str] = (),
    ) -> "Stream":
        """
        Parameters
        ----------
        batch_size: int
            Number of documents to process at a time in a GPU worker (or in the
            main process if no workers are used).
        batch_by: Literal["docs", "words", "padded_words"]
            How to compute the batch size. Can be "docs" or "words" :

            - "docs" (default) is the number of documents.
            - "words" is the total number of words in the documents.
            - "padded_words" is the total number of words in the documents, including
               padding, assuming the documents are padded to the same length.
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
            mode. If set to False, workers will process documents whenever they are
            available in a dynamic fashion, which may result in out-of-order processing.
            If set to true, tasks will be distributed in a static, round-robin fashion
            to workers. Defaults to True.

        Returns
        -------
        Stream
        """
        kwargs = {k: v for k, v in locals().items() if k in _non_default_args}
        if (
            kwargs.pop("chunk_size", INFER) is not INFER
            or kwargs.pop("sort_chunks", INFER) is not INFER
        ):
            warnings.warn(
                """chunk_size and sort_chunks are deprecated, use \
                map_batched(sort_fn, batch_size=chunk_size) instead.""",
                VisibleDeprecationWarning,
            )
        if kwargs.pop("split_into_batches_after", INFER) is not INFER:
            warnings.warn(
                "split_into_batches_after is deprecated.", VisibleDeprecationWarning
            )
        return Stream(
            reader=self.reader,
            writer=self.writer,
            ops=self.ops,
            config={
                **self.config,
                **{k: v for k, v in kwargs.items() if v is not INFER},
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
        Maps a callable to the documents.

        Parameters
        ----------
        pipe: Any
            The callable to map to the documents.
        name: Optional[str]
            The name of the pipeline step.
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
        batch_size: Optional[int] = None,
        batch_by: Optional[Union[str, Callable]] = None,
    ) -> "Stream":
        """
        Maps a callable to a batch of documents. The callable should take a list of
        inputs and return a **list** of outputs (not a single output).

        Parameters
        ----------
        pipe: Any
            The callable to map to the documents.
        name: Optional[str]
            The name of the pipeline step.
        kwargs: Dict
            The keyword arguments to pass to the callable.
        batch_size: Optional[int]
            The number of elements to process at a time in a GPU worker.
        batch_by: Optional[Union[str, Callable]]
            Function to compute the batches. If set, it should take an iterable of
            documents and return an iterable of batches. Defaults to "docs". You can
            also set it to "words" or "padded_words" to use predefined batching
            functions.

        Returns
        -------
        Stream
        """
        assert batch_by is None or batch_by in batchify_fns or callable(batch_by)
        batch_fn = batchify_fns.get(batch_by, batch_by)
        infer_batch = batch_size is None and batch_by is None
        ops = list(self.ops)
        if infer_batch and len(ops) and isinstance(ops[-1], UnbatchifyOp):
            ops.pop()
        else:
            ops.append(BatchifyOp(batch_size, batch_fn))
        ops.append(MapBatchesOp(Batchable(pipe), kwargs))
        ops.append(UnbatchifyOp())
        return Stream(
            reader=self.reader,
            writer=self.writer,
            ops=ops,
            config=self.config,
        )

    def batchify(
        self,
        batch_size: Optional[int] = None,
        batch_by: Optional[Union[str, Callable]] = None,
    ) -> "Stream":
        """
        Batches the documents.

        Parameters
        ----------
        batch_size: Optional[int]
            The number of elements to process at a time in a GPU worker.
        batch_by: Optional[Union[str, Callable]]
            Function to compute the batches. If set, it should take an iterable of
            documents and return an iterable of batches. Defaults to "docs". You can
            also set it to "words" or "padded_words" to use predefined batching
            functions.

        Returns
        -------
        Stream
        """
        assert batch_by is None or batch_by in batchify_fns or callable(batch_by)
        batch_fn = batchify_fns.get(batch_by, batch_by)
        ops = list(self.ops)
        ops.append(BatchifyOp(batch_size, batch_fn))
        return Stream(
            reader=self.reader,
            writer=self.writer,
            ops=ops,
            config=self.config,
        )

    def map_gpu(
        self,
        prepare_batch: Callable[[List, Union[str, torch.device]], Any],
        forward: Callable[[Any], Any],
        postprocess: Optional[Callable[[List, Any], Any]] = None,
        name: Optional[str] = None,
        batch_size: Optional[int] = None,
        batch_by: Optional[Union[str, Callable]] = None,
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
        name: Optional[str]
            The name of the pipeline step.
        batch_size: Optional[int]
            The number of elements to process at a time in a GPU worker.
        batch_by: Optional[Union[str, Callable]]
            Function to compute the batches. If set, it should take an iterable of
            documents and return an iterable of batches. Defaults to "docs". You can
            also set it to "words" or "padded_words" to use predefined batching
            functions.

        Returns
        -------
        Stream
        """
        assert batch_by is None or batch_by in batchify_fns or callable(batch_by)
        batch_fn = batchify_fns.get(batch_by, batch_by)
        infer_batch = batch_size is None and batch_by is None
        ops = list(self.ops)
        if infer_batch and len(ops) and isinstance(ops[-1], UnbatchifyOp):
            ops.pop()
        else:
            ops.append(BatchifyOp(batch_size, batch_fn))
        ops.append(GPUOp(prepare_batch, forward, postprocess))
        ops.append(UnbatchifyOp())
        return Stream(
            reader=self.reader,
            writer=self.writer,
            ops=ops,
            config=self.config,
        )

    def map_pipeline(
        self,
        model: Pipeline,
        batch_size: Optional[int] = INFER,
        batch_by: Optional[Union[str, Callable]] = INFER,
    ) -> "Stream":
        """
        Maps a pipeline to the documents.

        Parameters
        ----------
        model: Pipeline
            The pipeline to map to the documents.
        batch_size: Optional[int]
            The number of elements to process at a time in a GPU worker.
        batch_by: Optional[Union[str, Callable]]
            Function to compute the batches. If set, it should take an iterable of
            documents and return an iterable of batches. Defaults to "docs". You can
            also set it to "words" or "padded_words" to use predefined batching
            functions.

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
                    and "tokenizer" in inspect.signature(op.pipe).parameters
                    and "tokenizer" not in op.kwargs
                )
                or (
                    isinstance(op, MapBatchesOp)
                    and hasattr(op.pipe, "batch_process")
                    and "tokenizer"
                    in inspect.signature(op.pipe.batch_process).parameters
                    and "tokenizer" not in op.kwargs
                )
                or (
                    isinstance(op, MapBatchesOp)
                    and callable(op.pipe)
                    and "tokenizer" in inspect.signature(op.pipe).parameters
                    and "tokenizer" not in op.kwargs
                )
            ):
                op.kwargs["tokenizer"] = tokenizer
            new_ops.append(op)
        new_ops.append(MapOp(model._ensure_doc, {}))
        new_ops.append(BatchifyOp(batch_size, batchify_fns.get(batch_by, batch_by)))
        for name, pipe in model.pipeline:
            if name not in model._disabled:
                new_ops.append((MapBatchesOp(pipe, {})))
        new_ops.append(UnbatchifyOp())
        config = (
            {**self.config, "batch_size": model.batch_size}
            if self.batch_size is None
            else self.config
        )
        return Stream(
            reader=self.reader,
            writer=self.writer,
            ops=new_ops,
            config=config,
        )

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

    def copy(
        self,
        reader: bool = False,
        writer: bool = False,
        ops: bool = False,
        config: bool = False,
    ):
        return Stream(
            reader=copy(self.reader) if reader else self.reader,
            writer=copy(self.writer) if writer else self.writer,
            ops=copy(self.ops) if ops else self.ops,
            config=copy(self.config) if config else self.config,
        )

    def __dir__(self):  # pragma: no cover
        return (*super().__dir__(), *edsnlp.data.__all__)

    def __getattr__(self, item):
        return getattr(Stream, item).__get__(self)

    def _make_stages(
        self,
        split_torch_pipes: bool,
    ) -> List[Stage]:
        current_ops = []
        stages = []
        self_batch_fn = batchify_fns.get(self.batch_by, self.batch_by)
        self_batch_size = self.batch_size
        assert self_batch_size is not None

        for op in self.ops:
            op = copy(op)
            if isinstance(op, BatchifyOp):
                op.batch_fn = op.batch_fn or self_batch_fn
                op.size = op.size or self_batch_size
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

        return stages

    def __repr__(self):
        ops_str = ",\n".join(textwrap.indent(repr(op), "    ") for op in self.ops)
        return (
            f"Stream(\n"
            f"  reader={self.reader},\n"
            f"  ops=[\n{ops_str}\n  ],\n"
            f"  writer={self.writer})\n"
        )

    if TYPE_CHECKING:
        to_spark = edsnlp.data.to_spark  # noqa: F811
        to_pandas = edsnlp.data.to_pandas  # noqa: F811
        to_polars = edsnlp.data.to_polars  # noqa: F811
        to_iterable = edsnlp.data.to_iterable  # noqa: F811
        write_parquet = edsnlp.data.write_parquet  # noqa: F811
        write_standoff = edsnlp.data.write_standoff  # noqa: F811
        write_brat = edsnlp.data.write_brat  # noqa: F811
        write_json = edsnlp.data.write_json  # noqa: F811


# For backwards compatibility
LazyCollection = Stream
