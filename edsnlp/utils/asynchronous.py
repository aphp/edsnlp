import asyncio
from typing import Any, Coroutine, Optional, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Runs an asynchronous coroutine and always waits for the result,
    whether or not an event loop is already running.

    In a standard Python script (no active event loop), it uses `asyncio.run()`.
    In a notebook or environment with a running event loop, it applies a patch
    using `nest_asyncio` and runs the coroutine via `loop.run_until_complete`.

    Parameters
    ----------
    coro : Coroutine
        The coroutine to run.

    Returns
    -------
    T
        The result returned by the coroutine.
    """
    try:
        loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.get_running_loop().run_until_complete(coro)
    else:
        return asyncio.run(coro)
