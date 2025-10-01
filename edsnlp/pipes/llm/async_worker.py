import asyncio
import threading
from typing import Any, Coroutine, Dict, Iterable, Optional, Tuple


class AsyncRequestWorker:
    _instance = None

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._next_id = 0
        self._results: Dict[int, Tuple[Any, Optional[BaseException]]] = {}

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    @classmethod
    def instance(cls) -> "AsyncRequestWorker":
        if cls._instance is None:
            cls._instance = AsyncRequestWorker()
        return cls._instance

    def submit(self, coro: Coroutine[Any, Any, Any]) -> int:
        with self._lock:
            task_id = self._next_id
            self._next_id += 1

        async def _wrap():
            try:
                res = await coro
                exc = None
            except BaseException as e:  # noqa: BLE001
                res = None
                exc = e
            with self._cv:
                self._results[task_id] = (res, exc)
                self._cv.notify_all()

        asyncio.run_coroutine_threadsafe(_wrap(), self.loop)
        return task_id

    def pop_result(self, task_id: int) -> Optional[Tuple[Any, Optional[BaseException]]]:
        with self._cv:
            return self._results.pop(task_id, None)

    def wait_for_any(self, task_ids: Iterable[int]) -> int:
        task_ids = set(task_ids)
        with self._cv:
            while True:
                for tid in task_ids:
                    if tid in self._results:
                        return tid
                self._cv.wait()
