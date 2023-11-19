from __future__ import annotations

import contextlib
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import edsnlp.data

if TYPE_CHECKING:
    import torch

    from edsnlp import Pipeline
    from edsnlp.core.torch_component import TorchComponent
    from edsnlp.data.base import BaseReader, BaseWriter

INFER = type("INFER", (), {"__repr__": lambda self: "INFER"})()


class LazyCollection:
    def __init__(
        self,
        reader: Optional[BaseReader] = None,
        writer: Optional[BaseWriter] = None,
        pipeline: List[Any] = [],
        config={},
    ):
        self.reader = reader
        self.writer = writer
        self.pipeline: List[Tuple[str, Callable, Dict, Any]] = pipeline
        self.config = config

    @classmethod
    def ensure_lazy(cls, data):
        from edsnlp.data.base import IterableReader

        if isinstance(data, cls):
            return data
        return cls(reader=IterableReader(data))

    def map(self, pipe, name: Optional[str] = None, kwargs={}) -> "LazyCollection":
        return LazyCollection(
            reader=self.reader,
            writer=self.writer,
            pipeline=[*self.pipeline, (name, pipe, kwargs, None)],
            config=self.config,
        )

    def map_model(self, model: Pipeline) -> "LazyCollection":
        new_steps = []
        tokenizer = model.tokenizer
        for name, pipe, kwargs, pipe_tokenizer in self.pipeline:
            new_steps.append((name, pipe, kwargs, pipe_tokenizer or tokenizer))
        new_steps.append(("_ensure_doc", model._ensure_doc, {}, tokenizer))
        for name, pipe in model.pipeline:
            if name not in model._disabled:
                new_steps.append((name, pipe, {}, tokenizer))
        config = (
            {**self.config, "batch_size": model.batch_size}
            if self.batch_size is None
            else self.config
        )
        return LazyCollection(
            reader=self.reader,
            writer=self.writer,
            pipeline=new_steps,
            config=config,
        )

    def write(self, writer: BaseWriter, execute: bool = True) -> Any:
        lc = LazyCollection(
            reader=self.reader,
            writer=writer,
            pipeline=self.pipeline,
            config=self.config,
        )
        return lc.execute() if execute else lc

    def execute(self):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self.execute())

    @contextlib.contextmanager
    def cache(self):
        for name, pipe, *_ in self.pipeline:
            if hasattr(pipe, "enable_cache"):
                pipe.enable_cache()
        yield
        for name, pipe, *_ in self.pipeline:
            if hasattr(pipe, "disable_cache"):
                pipe.disable_cache()

    def torch_components(
        self, disable: Container[str] = ()
    ) -> Iterable[Tuple[str, "TorchComponent"]]:
        """
        Yields components that are PyTorch modules.

        Parameters
        ----------
        disable: Container[str]
            The names of disabled components, which will be skipped.

        Returns
        -------
        Iterable[Tuple[str, 'edsnlp.core.torch_component.TorchComponent']]
        """
        for name, pipe, *_ in self.pipeline:
            if name not in disable and hasattr(pipe, "batch_process"):
                yield name, pipe

    def to(self, device: Union[str, Optional["torch.device"]] = None):  # noqa F821
        """Moves the pipeline to a given device"""
        for name, pipe, *_ in self.torch_components():
            pipe.to(device)
        return self

    def worker_copy(self):
        return LazyCollection(
            reader=self.reader.worker_copy(),
            writer=self.writer,
            pipeline=self.pipeline,
            config=self.config,
        )


if TYPE_CHECKING:
    # just to add read/from_* and write/to_* methods to the static type hints
    LazyCollection = edsnlp.data  # noqa: F811
