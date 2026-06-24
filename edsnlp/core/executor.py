from __future__ import annotations

import asyncio
import copy
import threading
from concurrent.futures import Future
from itertools import count
from typing import TYPE_CHECKING

from edsnlp.core.stream import StreamRecord
from edsnlp.data.base import QueueReader
from edsnlp.utils.batching import BatchTimeoutSentinel
from edsnlp.utils.stream_sentinels import StreamSentinel

if TYPE_CHECKING:
    from pydantic import NonNegativeFloat

    from edsnlp.core.stream import Stream


def set_future_result(future, value):
    if not future.done():
        future.set_result(value)


def set_future_exception(future, exc):
    if not future.done():
        future.set_exception(exc)


async def wait_thread_future(future):
    try:
        return await asyncio.wrap_future(future)
    except asyncio.CancelledError:  # pragma: no cover
        future.cancel()
        raise


class StreamExecutor:
    def __init__(
        self,
        stream: "Stream",
        batch_wait_timeout: "NonNegativeFloat | None" = None,
    ):
        self.validate(stream)
        self.stream = stream
        reader = copy.copy(stream.reader)
        if batch_wait_timeout is not None:
            reader.batch_wait_timeout = batch_wait_timeout
        self.batch_wait_timeout = reader.batch_wait_timeout
        self.reader = reader
        self._task_ids = count()
        self._input_queue = reader.queue
        self._closed = False
        self._error = None
        self._lock = threading.Lock()
        self._futures = {}

        from edsnlp.core.stream import Stream

        runner_config = {
            **stream.config,
            "_executor_batch_wait_timeout": self.batch_wait_timeout,
            "_executor_unordered_outputs": True,
        }

        runner_stream = Stream(
            reader=reader,
            writer=stream.writer,
            ops=stream.ops,
            config=runner_config,
        )
        self.runner = runner_stream.execute()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, item):
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._closed:
                raise RuntimeError("Executor is closed")
            if self._error is not None:  # pragma: no cover
                raise self._error
            task_id = next(self._task_ids)
            future = Future()
            self._futures[task_id] = future
            self._input_queue.put(StreamRecord(task_id, item))
            return loop.create_task(wait_thread_future(future))

    def close(self):
        with self._lock:
            if self._closed:  # pragma: no cover
                return
            self._closed = True
            self.reader.close()
        self.thread.join()

    def run(self):
        try:
            for item in self.runner:
                if isinstance(item, BatchTimeoutSentinel):  # pragma: no cover
                    continue
                if isinstance(item, StreamSentinel):  # pragma: no cover
                    continue
                if not isinstance(item, StreamRecord):
                    raise TypeError("Executor stream must preserve stream records")
                self.resolve(item)
        except BaseException as exc:  # noqa: BLE001  # pragma: no cover
            self._error = exc
            self.fail_pending(exc)
        else:
            self.fail_pending(RuntimeError("Executor closed"))

    def resolve(self, item):
        with self._lock:
            future = self._futures.pop(item.id, None)
        if future is not None:
            value = item.value
            if item.error is not None:
                set_future_exception(future, item.error)
            else:
                set_future_result(future, value)

    def fail_pending(self, exc):
        with self._lock:
            futures = list(self._futures.values())
            self._futures = {}
        for future in futures:  # pragma: no cover
            set_future_exception(future, exc)

    async def aclose(self) -> None:
        self.close()

    async def __aenter__(self) -> "StreamExecutor":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    @staticmethod
    def validate(stream: "Stream") -> None:
        if not isinstance(stream.reader, QueueReader):
            raise ValueError(
                "Executors require a stream built with edsnlp.data.from_queue"
            )
        if stream.writer is not None:
            raise ValueError("Executors do not support writers")
        if stream.backend == "spark":
            raise ValueError("Executors do not support the spark backend")
        if any(not op.elementwise for op in stream.ops):
            raise ValueError("Executors require an elementwise stream")
