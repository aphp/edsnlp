from __future__ import annotations

import copyreg
import io
import logging
import math
import multiprocessing
import multiprocessing.reduction
import os
import sys
import tempfile
import threading
import warnings
from itertools import chain, cycle, islice
from multiprocessing.connection import wait
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import dill
from tqdm import tqdm
from typing_extensions import TypedDict

from edsnlp.core.lazy_collection import LazyCollection
from edsnlp.data.base import BaseReader, BaseWriter, BatchWriter
from edsnlp.utils.collections import (
    batch_compress_dict,
    decompress_dict,
)

doc_size_fns = {
    "words": len,
}

if TYPE_CHECKING:
    import torch

Stage = TypedDict(
    "Stage",
    {
        "cpu_components": List[Tuple[str, Callable, Dict, Any]],
        "gpu_component": Optional[Any],
    },
)


class StopType:
    # Singleton is important since the STOP object may be passed to
    # other processes, i.e. pickled, depickled, while it should
    # always be the same object.
    instance = None

    def __repr__(self):
        return "STOP"

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance


STOP = StopType()


class ForkingPickler(dill.Pickler):
    """
    ForkingPickler that uses dill instead of pickle to transfer objects between
    processes.
    """

    _extra_reducers = {}
    _copyreg_dispatch_table = copyreg.dispatch_table

    def __new__(cls, *args, **kwargs):
        result = dill.Pickler.__new__(ForkingPickler)
        # Python would not call __init__ if the original
        # multiprocessing.reduction.ForkingPickler called, leading to a call to this
        # monkey-patched __new__ method, because [original cls] != [type of result]
        # (see https://docs.python.org/3/reference/datamodel.html#basic-customization)
        # so we force the call to __init__ here
        if not isinstance(result, cls):
            result.__init__(*args, **kwargs)
        return result

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.dispatch_table = self._copyreg_dispatch_table.copy()
        self.dispatch_table.update(self._extra_reducers)

    @classmethod
    def register(cls, type, reduce):
        """Register a reduce function for a type."""
        cls._extra_reducers[type] = reduce

    @classmethod
    def dumps(cls, obj, protocol=None, *args, **kwds):
        buf = io.BytesIO()
        cls(buf, protocol, *args, **kwds).dump(obj)
        return buf.getbuffer()

    loads = dill.loads


def replace_pickler():
    """
    Replace the default pickler used by multiprocessing with dill.
    "multiprocess" didn't work for obscure reasons (maybe the reducers / dispatchers
    are not propagated between multiprocessing and multiprocess => torch specific
    reducers might be missing ?), so this patches multiprocessing directly.
    directly.

    For some reason I do not explain, this has a massive impact on the performance of
    the multiprocessing backend. With the original pickler, the performance can be
    up to 2x slower than with our custom one.
    """
    old_pickler = multiprocessing.reduction.ForkingPickler

    before = (
        dict(ForkingPickler._extra_reducers),
        old_pickler.__new__,
        old_pickler.dumps,
        old_pickler.loads,
        old_pickler.register,
    )

    old_pickler.__new__ = ForkingPickler.__new__
    old_pickler.dumps = ForkingPickler.dumps
    old_pickler.loads = ForkingPickler.loads
    old_pickler.register = ForkingPickler.register
    ForkingPickler._extra_reducers.update(
        multiprocessing.reduction.ForkingPickler._extra_reducers  # noqa
    )

    def revert():
        (
            ForkingPickler._extra_reducers,
            old_pickler.__new__,
            old_pickler.dumps,
            old_pickler.loads,
            old_pickler.register,
        ) = before

    return revert


def cpu_count():  # pragma: no cover
    """
    Heavily inspired (partially copied) from joblib's loky
    (https://github.com/joblib/loky/blob/2c21e/loky/backend/context.py#L83)
    by Thomas Moreau and Olivier Grisel.

    Return the number of CPUs we can use to process data in parallel.

    The returned number of CPUs returns the minimum of:
     * `os.cpu_count()`
     * the CPU affinity settings
     * cgroup CPU bandwidth limit (share of total CPU time allowed in a given job)
       typically used in containerized environments like Docker

    Note that on Windows, the returned number of CPUs cannot exceed 61 (or 60 for
    Python < 3.10), see:
    https://bugs.python.org/issue26903.

    It is also always larger or equal to 1.
    """
    # Note: os.cpu_count() is allowed to return None in its docstring
    os_cpu_count = os.cpu_count() or 1
    if sys.platform == "win32":
        # Following loky's windows implementation

        _MAX_WINDOWS_WORKERS = 60
        if sys.version_info >= (3, 8):
            from concurrent.futures.process import _MAX_WINDOWS_WORKERS  # noqa

            if sys.version_info < (3, 10):
                _MAX_WINDOWS_WORKERS = _MAX_WINDOWS_WORKERS - 1
        os_cpu_count = min(os_cpu_count, _MAX_WINDOWS_WORKERS)

    cpu_count_affinity = os_cpu_count
    try:
        cpu_count_affinity = len(os.sched_getaffinity(0))
    except (NotImplementedError, AttributeError):
        pass

    # Cgroup CPU bandwidth limit available in Linux since 2.6 kernel
    cpu_count_cgroup = os_cpu_count
    cpu_max_fname = "/sys/fs/cgroup/cpu.max"
    cfs_quota_fname = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    cfs_period_fname = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    if os.path.exists(cpu_max_fname):
        # cgroup v2
        # https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
        with open(cpu_max_fname) as fh:
            cpu_quota_us, cpu_period_us = fh.read().strip().split()
    elif os.path.exists(cfs_quota_fname) and os.path.exists(cfs_period_fname):
        # cgroup v1
        # https://www.kernel.org/doc/html/latest/scheduler/sched-bwc.html#management
        with open(cfs_quota_fname) as fh:
            cpu_quota_us = fh.read().strip()
        with open(cfs_period_fname) as fh:
            cpu_period_us = fh.read().strip()
    else:
        cpu_quota_us = "max"
        cpu_period_us = 100_000

    if cpu_quota_us != "max":
        cpu_quota_us = int(cpu_quota_us)
        cpu_period_us = int(cpu_period_us)
        if cpu_quota_us > 0 and cpu_period_us > 0:
            cpu_count_cgroup = math.ceil(cpu_quota_us / cpu_period_us)

    return max(1, min(os_cpu_count, cpu_count_affinity, cpu_count_cgroup))


# Should we check if the multiprocessing module of edsnlp
# is responsible for this child process before replacing the pickler ?
if (
    multiprocessing.current_process() != "MainProcess"
    or hasattr(multiprocessing, "parent_process")
    and multiprocessing.parent_process() is not None
):
    replace_pickler()

DEBUG = True

debug = (
    (lambda *args, flush=False, **kwargs: print(*args, **kwargs, flush=True))
    if DEBUG
    else lambda *args, **kwargs: None
)

if os.environ.get("TORCH_SHARING_STRATEGY"):
    try:
        torch.multiprocessing.set_sharing_strategy(os.environ["TORCH_SHARING_STRATEGY"])
    except NameError:
        pass

try:
    import torch

    # Torch may still be imported as a namespace package, so we can access the
    # torch.save and torch.load functions
    torch_save = torch.save
    torch_load = torch.load

    MAP_LOCATION = None

    try:
        from accelerate.hooks import AlignDevicesHook

        # We need to replace the "execution_device" attribute of the AlignDevicesHook
        # using map_location when unpickling the lazy collection

        def save_align_devices_hook(pickler: Any, obj: Any):
            pickler.save_reduce(load_align_devices_hook, (obj.__dict__,), obj=obj)

        def load_align_devices_hook(state):
            state["execution_device"] = MAP_LOCATION
            new_obj = AlignDevicesHook.__new__(AlignDevicesHook)
            new_obj.__dict__.update(state)
            return new_obj

    except ImportError:
        AlignDevicesHook = None

    def dump(*args, **kwargs):
        # We need to replace the "execution_device" attribute of the AlignDevicesHook
        # using map_location when pickling the lazy collection
        old = None
        try:
            if AlignDevicesHook is not None:
                old = dill.Pickler.dispatch.get(AlignDevicesHook)
                dill.Pickler.dispatch[AlignDevicesHook] = save_align_devices_hook
            dill.settings["recurse"] = True
            return torch_save(*args, pickle_module=dill, **kwargs)
        finally:
            dill.settings["recurse"] = False
            if AlignDevicesHook is not None:
                del dill.Pickler.dispatch[AlignDevicesHook]
                if old is not None:  # pragma: no cover
                    dill.Pickler.dispatch[AlignDevicesHook] = old

    def load(*args, map_location=None, **kwargs):
        global MAP_LOCATION
        MAP_LOCATION = map_location
        if torch.__version__ >= "2.1" and isinstance(args[0], str):
            kwargs["mmap"] = True
        try:
            if torch.__version__ < "2.0.0":
                torch_load.__globals__["pickle"] = dill
            result = torch_load(
                *args,
                pickle_module=dill,
                map_location=map_location,
                **kwargs,
            )
        finally:
            import pickle

            torch_load.__globals__["pickle"] = pickle
        MAP_LOCATION = None
        return result

except (ImportError, AttributeError):  # pragma: no cover
    # noinspection PyUnusedLocal
    def load(file, *args, map_location=None, **kwargs):
        # check if path
        if isinstance(file, str):
            with open(file, "rb") as f:
                return dill.load(f, *args, **kwargs)
        return dill.load(file, *args, **kwargs)

    dump = dill.dump


def get_dispatch_schedule(cpu_idx: int, cpus: range, gpus: range) -> List[int]:
    """
    To which GPU worker should a given CPU worker dispatch its data to. This function
    returns a list of GPU workers over a period to be determined by the function.
    This is actually a fun problem, because we want:
    - to distribute the data evenly between the GPU workers
    - minimize the number of distinct unique CPU workers sending data to the same GPU
      (because we move the tensors to the GPU inside the CPU worker, which
       creates takes a bit of VRAM for each CPU worker)

    Parameters
    ----------
    cpu_idx: int
        Index of the CPU worker
    cpus: range
        Range of CPU workers
    gpus: range
        Range of GPU workers

    Returns
    -------
    List[int]
    """
    idx = cpus.index(cpu_idx)
    base = []

    if gpus:
        if len(gpus) < len(cpus):
            r = len(cpus) % len(gpus)  # number of CPU without an easy to assign GPU
            R = cpus[len(cpus) - r :]
            if cpu_idx not in R:
                base = [gpus[idx % len(gpus)]]
                base = base * (len(gpus) // len(base))
            else:
                base = get_dispatch_schedule(cpu_idx, R, gpus)
        else:
            idx = cpus.index(cpu_idx)
            r = len(gpus) % len(cpus)
            d = len(gpus) // len(cpus)
            base = list(gpus[idx * d : (idx + 1) * d])
            base = base * len(cpus)  # (len(G) // len(G))
            R = gpus[-r:]
            if r > 0:
                base = base + get_dispatch_schedule(cpu_idx, cpus, R)
    return base


class CPUWorker:
    def __init__(
        self,
        cpu_idx: int,
        exchanger: Exchanger,
        lazy_collection_path: str,
        device: Union[str, "torch.device"],
    ):
        super(CPUWorker, self).__init__()

        self.cpu_idx = cpu_idx
        self.exchanger = exchanger
        self.lazy_collection_path = lazy_collection_path
        self.device = device

    def run(self):
        def handle_stage(stage_idx):
            nonlocal reader, writer

            if reader.read_in_worker and stage_idx == 0:

                def dequeue():
                    items = iter(reader.read_records())
                    items = islice(items, self.cpu_idx, None, num_cpu_workers)
                    yield from items
            else:

                def dequeue():
                    num_prod_alive = self.exchanger.num_producers_for_cpu(stage_idx)

                    while num_prod_alive > 0:
                        item = self.exchanger.get_cpu(self.cpu_idx, stage_idx)
                        if item is STOP:
                            num_prod_alive -= 1
                            continue
                        yield item

            try:
                gpu_queue_schedule = iter(
                    cycle(
                        get_dispatch_schedule(
                            self.cpu_idx,
                            range(self.exchanger.num_cpu_workers),
                            range(self.exchanger.num_gpu_workers),
                        )
                    )
                )

                # Get items from the previous stage
                items = dequeue()

                last_torch_pipe = (
                    getattr(stages[stage_idx - 1], "gpu_op", None)
                    if stage_idx > 0
                    else None
                )
                if hasattr(last_torch_pipe, "postprocess"):

                    def pipe_postprocess(items):
                        x = active_batches[stage_idx - 1]
                        for batch_id, result in items:
                            slot = next(s for s in x if s[0] == batch_id)
                            _, docs, inputs, _ = slot
                            docs = last_torch_pipe.postprocess(docs, result, inputs)
                            slot[3] = docs
                            if lc.deterministic:
                                while x and x[0][3] is not None:
                                    yield x.pop(0)[3]
                            else:
                                x.remove(slot)
                                yield docs

                    items = pipe_postprocess(items)

                stage = stages[stage_idx]
                for op in stage.cpu_ops:
                    items = op(items)
                if stage.gpu_op is not None:
                    torch_pipe = stage.gpu_op
                    for docs in items:
                        batch_id = hash(tuple(id(x) for x in docs))
                        torch_pipe.enable_cache(batch_id)
                        inputs = [torch_pipe.preprocess(x) for x in docs]
                        batch = decompress_dict(list(batch_compress_dict(inputs)))
                        batch = torch_pipe.collate(batch)
                        gpu_idx = next(gpu_queue_schedule)
                        device = self.exchanger.gpu_worker_devices[gpu_idx]
                        batch = torch_pipe.batch_to_device(batch, device)
                        active_batches[stage_idx].append([batch_id, docs, inputs, None])
                        self.exchanger.send_to_gpu_worker(
                            item=(self.cpu_idx, batch_id, batch),
                            idx=gpu_idx,
                            stage=stage_idx,
                        )
                        if stage_idx == len(stages) - 2:
                            torch_pipe.disable_cache(batch_id)
                else:
                    if writer is not None:
                        items = (writer.handle_record(rec) for rec in items)
                    if getattr(writer, "batch_in_worker", None) is True:
                        items = writer.batch_by(items, writer.batch_size)
                        items = (writer.handle_batch(b) for b in items)
                    else:
                        items = ((x, 1) for x in items)
                    for out in items:
                        self.exchanger.put_results(out, self.cpu_idx)

                self.exchanger.notify_cpu(
                    "DONE",
                    self.cpu_idx,
                    stage_idx,
                )
                self.exchanger.stop_consumers_from_cpu(self.cpu_idx, stage_idx)
            except BaseException as e:
                import traceback

                print(f"Error in {self}:\n{traceback.format_exc()}", flush=True)
                self.exchanger.notify_cpu("STOP", self.cpu_idx, None)
                self.exchanger.notify_main(e)

        # Cannot pass torch tensor during init i think ? otherwise i get
        # ValueError: bad value(s) in fds_to_keep
        # mp._prctl_pr_set_pdeathsig(signal.SIGINT)
        try:
            lc: LazyCollection = load(
                self.lazy_collection_path, map_location=self.device
            )
            reader = lc.reader
            writer: Union[BaseWriter, BatchWriter] = lc.writer
            num_cpu_workers = self.exchanger.num_cpu_workers
            lc.eval()
            stages = lc._make_stages(len(self.exchanger.gpu_worker_devices) > 0)
            active_batches = [[] for _ in range(len(stages) - 1)]

            # Inform the main process that we are ready
            self.exchanger.notify_main("READY")

            stage_threads = []
            alive_threads = set()
            for stage_idx in range(len(stages)):
                thread = threading.Thread(
                    target=handle_stage,
                    args=(stage_idx,),
                    daemon=True,
                    name=f"EDSNLP-CPUWorker-{self.cpu_idx}-stage-{stage_idx}",
                )
                thread.start()
                alive_threads.add(stage_idx)
                stage_threads.append(thread)

            while alive_threads:
                msg, stage_idx = self.exchanger.get_cpu_notification(self.cpu_idx)
                if msg == "STOP":
                    return
                if msg == "DONE":
                    alive_threads.remove(stage_idx)
                    continue

            import edsnlp.core.torch_component

            assert list(edsnlp.core.torch_component._caches) == []

            for thread in stage_threads:
                thread.join()

        except BaseException as e:  # pragma: no cover
            import traceback

            print(f"Error in {self}:\n{traceback.format_exc()}", flush=True)
            self.exchanger.notify_main(e)

    def __repr__(self):
        return f"<CPUWorker idx={self.cpu_idx}>"


class GPUWorker:
    def __init__(
        self,
        gpu_idx: int,
        exchanger: Exchanger,
        lazy_collection_path: str,
        device: Union[str, "torch.device"],
    ):
        super(GPUWorker, self).__init__()

        self.gpu_idx = gpu_idx
        self.exchanger = exchanger
        self.lazy_collection_path = lazy_collection_path
        self.device = device

    def run(self):
        def handle_stage(stage_idx):
            num_prod_alive = self.exchanger.num_producers_for_gpu(stage_idx)
            cpu_idx = 0

            try:
                # Get items from the previous stage
                while num_prod_alive > 0:
                    item = self.exchanger.get_gpu_task(self.gpu_idx, stage_idx)
                    if item is STOP:
                        num_prod_alive -= 1
                        continue

                    cpu_idx, batch_id, batch = item

                    pipe = stages[stage_idx].gpu_op
                    pipe.enable_cache(batch_id)
                    batch = pipe(batch)
                    batch = {
                        k: v.to("cpu") if hasattr(v, "to") else v
                        for k, v in batch.items()
                    }
                    self.exchanger.send_to_cpu_worker(
                        item=(batch_id, batch),
                        idx=cpu_idx,
                        stage_idx=stage_idx + 1,
                    )
                    del batch
                    if stage_idx == len(stages) - 2:  # Last GPU stage
                        pipe.disable_cache(batch_id)

                self.exchanger.notify_gpu("DONE", self.gpu_idx, stage_idx)
                self.exchanger.stop_cpu_consumers(stage_idx + 1)
            except BaseException as e:
                import traceback

                print(f"Error in {self}:\n{traceback.format_exc()}", flush=True)
                self.exchanger.notify_gpu("STOP", self.gpu_idx, None)
                self.exchanger.notify_main(e)

        try:
            lc: LazyCollection = load(
                self.lazy_collection_path, map_location=self.device
            )
            lc.eval()
            stages = lc._make_stages(len(self.exchanger.gpu_worker_devices) > 0)

            # Inform the main process that we are ready
            self.exchanger.notify_main("READY")

            stage_threads = []
            alive_threads = set()
            for stage_idx in range(len(stages) - 1):
                thread = threading.Thread(
                    target=handle_stage,
                    args=(stage_idx,),
                    daemon=True,
                    name=f"EDSNLP-GPUWorker-{self.gpu_idx}-stage-{stage_idx}",
                )
                thread.start()
                alive_threads.add(stage_idx)
                stage_threads.append(thread)

            while alive_threads:
                msg, stage_idx = self.exchanger.get_gpu_notification(self.gpu_idx)
                if msg == "STOP":
                    return
                if msg == "DONE":
                    alive_threads.remove(stage_idx)
                    continue
            import edsnlp.core.torch_component

            assert list(edsnlp.core.torch_component._caches) == []
            for thread in stage_threads:
                thread.join()

        except BaseException as e:  # pragma: no cover
            import traceback

            print(f"Error in {self}:\n{traceback.format_exc()}", flush=True)
            self.exchanger.notify_main(e)

    def __repr__(self):
        return f"<GPUWorker idx={self.gpu_idx}>"


class Exchanger:
    def __init__(
        self,
        mp: multiprocessing.context.BaseContext,
        num_cpu_workers: int,
        num_gpu_workers: int,
        num_stages: int,
        gpu_worker_devices: List[Any],
        share_input_queue: bool = False,
        share_output_queue: bool = False,
    ):
        self.gpu_worker_devices = gpu_worker_devices
        self.num_cpu_workers = num_cpu_workers
        self.num_gpu_workers = num_gpu_workers
        self.num_stages = num_stages
        input_queue = mp.Queue() if share_input_queue else None
        self._output_queues = (
            [mp.Queue(maxsize=4)] * num_cpu_workers
            if share_output_queue
            else [mp.Queue(maxsize=4) for _ in range(num_cpu_workers)]
        )
        self._cpu_queues = [
            ([input_queue] if share_input_queue else [mp.Queue()])
            + [mp.Queue(maxsize=2) for _ in range(num_stages - 1)]
            for _ in range(num_cpu_workers)
        ]
        self._gpu_queues = [
            [mp.Queue(maxsize=2) for _ in range(num_stages - 1)]
            for _ in range(num_gpu_workers)
        ]
        self._cpu_notify_queues = [mp.Queue() for _ in range(num_cpu_workers)]
        self._gpu_notify_queues = [mp.Queue() for _ in range(num_gpu_workers)]
        self._main_notify_queue = mp.Queue()

    def get_output_or_error(self, idx):
        ready = wait(
            [self._output_queues[idx]._reader, self._main_notify_queue._reader]
        )
        if self._main_notify_queue._reader in ready:
            raise self._main_notify_queue.get()
        return self._output_queues[idx].get()

    def send_input(self, item, idx: int, timeout):
        self._cpu_queues[idx][0].put(item, timeout=timeout)

    def put_results(self, item, idx):
        self._output_queues[idx].put(item)

    def get_cpu(self, idx: int, stage: int, nowait=False):
        return (
            self._cpu_queues[idx][stage].get()
            if not nowait
            else self._cpu_queues[idx][stage].get_nowait()
        )

    def send_to_cpu_worker(self, item, idx: int, stage_idx: int):
        self._cpu_queues[idx][stage_idx].put(item)

    def get_gpu_task(self, idx: int, stage: int):
        return self._gpu_queues[idx][stage].get()

    def send_to_gpu_worker(self, item, idx: int, stage: int):
        self._gpu_queues[idx][stage].put(item)

    def stop_consumers_from_cpu(self, cpu_idx, stage_idx):
        if stage_idx == self.num_stages - 1:
            self._output_queues[cpu_idx].put(STOP)
        else:
            for queue in self._gpu_queues:
                queue[stage_idx].put(STOP)

    def stop_cpu_consumers(self, stage_idx):
        for queue in self._cpu_queues:
            queue[stage_idx].put(STOP)

    def num_producers_for_cpu(self, stage_idx):
        return 1 if stage_idx == 0 else self.num_gpu_workers

    def num_producers_for_gpu(self, stage_idx):
        return self.num_cpu_workers

    def stop_everything(self):
        for queues in (*self._cpu_queues, *self._gpu_queues, self._output_queues):
            for queue in queues:
                if not queue._closed:
                    try:
                        while True:
                            queue.get_nowait()
                    except multiprocessing.queues.Empty:  # pragma: no cover
                        pass
                    try:
                        queue.put_nowait(STOP)
                    except multiprocessing.queues.Full:  # pragma: no cover
                        pass
        for queue in (
            *self._cpu_notify_queues,
            *self._gpu_notify_queues,
        ):
            if not queue._closed:
                queue.put_nowait(("STOP", None))
        for q in (
            *self._output_queues,
            *chain.from_iterable(self._cpu_queues),
            *chain.from_iterable(self._gpu_queues),
            *self._cpu_notify_queues,
            *self._gpu_notify_queues,
            self._main_notify_queue,
        ):
            q.close()
        assert all(
            q._closed
            for q in (
                *self._output_queues,
                *chain.from_iterable(self._cpu_queues),
                *chain.from_iterable(self._gpu_queues),
                *self._cpu_notify_queues,
                *self._gpu_notify_queues,
                self._main_notify_queue,
            )
        )

    def notify_cpu(self, msg, cpu_idx, stage_idx):
        self._cpu_notify_queues[cpu_idx].put_nowait((msg, stage_idx))

    def notify_gpu(self, msg, gpu_idx, stage_idx):
        self._gpu_notify_queues[gpu_idx].put_nowait((msg, stage_idx))

    def notify_main(self, msg):
        self._main_notify_queue.put_nowait(msg)

    def get_cpu_notification(self, cpu_idx):
        return self._cpu_notify_queues[cpu_idx].get()

    def get_gpu_notification(self, gpu_idx):
        return self._gpu_notify_queues[gpu_idx].get()

    def get_main_notification(self):
        return self._main_notify_queue.get()


def execute_multiprocessing_backend(
    lc: LazyCollection,
):
    """
    If you have multiple CPU cores, and optionally multiple GPUs, we provide the
    `multiprocessing` backend that allows to run the inference on multiple
    processes.

    This accelerator dispatches the batches between multiple workers
    (data-parallelism), and distribute the computation of a given batch on one or two
    workers (model-parallelism):

    - a `CPUWorker` which handles the non deep-learning components and the
      preprocessing, collating and postprocessing of deep-learning components
    - a `GPUWorker` which handles the forward call of the deep-learning components

    If no GPU is available, no `GPUWorker` is started, and the `CPUWorkers` handle
    the forward call of the deep-learning components as well.

    The advantage of dedicating a worker to the deep-learning components is that it
    allows to prepare multiple batches in parallel in multiple `CPUWorker`, and ensure
    that the `GPUWorker` never wait for a batch to be ready.

    The overall architecture described in the following figure, for 3 CPU workers and 2
    GPU workers.

    <div style="text-align:center">
        <img src="/assets/images/multiprocessing.png" style="height:400px" />
    </div>

    Here is how a small pipeline with rule-based components and deep-learning components
    is distributed between the workers:

    <div style="text-align:center">
        <img src="/assets/images/model-parallelism.png" />
    </div>

    !!! warning "Caveat"

        Since workers can produce their results in any order, the order of the results
        may not be the same as the order of the input tasks.
    """
    global glob_workers
    (
        num_cpu_workers,
        num_gpu_workers,
        cpu_worker_devices,
        gpu_worker_devices,
        has_torch_pipes,
    ) = adjust_num_workers(lc)
    stages = lc._make_stages(split_torch_pipes=num_gpu_workers > 0)
    reader: BaseReader = lc.reader
    writer: BaseWriter = lc.writer
    mp = get_multiprocessing_context(has_torch_pipes, lc.process_start_method)

    # Queues definition
    exchanger = Exchanger(
        mp=mp,
        num_cpu_workers=num_cpu_workers,
        num_gpu_workers=num_gpu_workers,
        num_stages=len(stages),
        gpu_worker_devices=gpu_worker_devices,
        share_input_queue=not lc.deterministic,
        share_output_queue=not lc.deterministic,
    )

    lc_to_dump = lc.worker_copy()
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        dump(lc_to_dump, fp)
        fp.close()

    del lc_to_dump

    revert_pickler = replace_pickler()
    revert_environ = setup_environ(lc.disable_implicit_parallelism)

    cpu_workers = []
    for cpu_idx in range(num_cpu_workers):
        worker = CPUWorker(
            cpu_idx=cpu_idx,
            exchanger=exchanger,
            lazy_collection_path=fp.name,
            device=cpu_worker_devices[cpu_idx],
        )
        cpu_workers.append(
            mp.Process(target=CPUWorker.run, args=(worker,), daemon=True)
        )

    gpu_workers = []
    for gpu_idx in range(num_gpu_workers):
        worker = GPUWorker(
            gpu_idx=gpu_idx,
            exchanger=exchanger,
            lazy_collection_path=fp.name,
            device=gpu_worker_devices[gpu_idx],
        )
        gpu_workers.append(
            mp.Process(target=GPUWorker.run, args=(worker,), daemon=True)
        )

    logging.debug(f"Main PID {os.getpid()}")
    logging.debug(
        f"Starting {num_cpu_workers} cpu workers and {num_gpu_workers} gpu workers on "
        f"{gpu_worker_devices} in {mp.get_start_method()} mode to run {len(stages)} "
        f"stage{'s' if len(stages) > 1 else ''}.",
    )

    for worker in (*cpu_workers, *gpu_workers):
        glob_workers.append(worker)
        worker.start()

    for _ in range(len(cpu_workers) + len(gpu_workers)):
        outputs = exchanger.get_main_notification()

        if isinstance(outputs, BaseException):
            raise outputs

    logging.debug("Workers are ready")

    os.unlink(fp.name)

    bar = tqdm(smoothing=0.1, mininterval=1.0, disable=not lc.show_progress)

    keep_going = True

    def stop():
        nonlocal keep_going
        keep_going = False
        revert_pickler()
        revert_environ()

        # Send gpu and cpu process the order to stop processing data
        # We use the prioritized queue to ensure the stop signal is processed
        # before the next batch of data
        exchanger.stop_everything()
        for worker in (*cpu_workers, *gpu_workers):
            if not worker._closed:
                try:
                    worker.join(10)
                except multiprocessing.TimeoutError:  # pragma: no cover
                    worker.kill()
                    print("Killed worker", worker, flush=True)
                worker.close()
            assert worker._popen is None
        assert all(w._closed for w in (*cpu_workers, *gpu_workers))

    def enqueue_inputs():
        try:
            items = iter(reader.read_records())
            # TODO: handle WORLD_SIZE env vars here
            # items = islice(items, self.cpu_idx, None, lc.num_cpu_workers)
            item = None
            last_item_sent = True
            worker_idx = 0
            while keep_going:
                if last_item_sent:
                    try:
                        item = next(items)
                        last_item_sent = False
                    except StopIteration:
                        exchanger.stop_cpu_consumers(0)
                        break
                try:
                    # worker_idx doesn't matter if share_output_queue is True
                    exchanger.send_input(item, worker_idx, timeout=1.0)
                    worker_idx = (worker_idx + 1) % num_cpu_workers
                    last_item_sent = True

                except multiprocessing.queues.Full:  # pragma: no cover
                    continue
        except BaseException as e:
            exchanger.notify_main(e)
            raise e

    def dequeue_outputs():
        try:
            with bar:
                active_workers = [True] * num_cpu_workers
                worker_idx = -1
                while keep_going and sum(active_workers) > 0:
                    worker_idx = (worker_idx + 1) % len(active_workers)
                    if active_workers[worker_idx] is False:
                        continue
                    out = exchanger.get_output_or_error(worker_idx)
                    # worker_idx doesn't matter if share_input_queue is True
                    if out is STOP:
                        active_workers[worker_idx] = False
                        continue
                    item, count = out
                    yield item
                    bar.update(count)
        finally:
            stop()

    # If we don't read the data in the worker, then each worker will have its own copy
    # of the data (usually file paths in this mode), so we don't need to send the data
    if not reader.read_in_worker:
        enqueue_inputs_thread = threading.Thread(
            target=enqueue_inputs,
            name="EDSNLP-enqueue-inputs",
            daemon=True,
        )
        enqueue_inputs_thread.start()

    items = dequeue_outputs()
    if getattr(writer, "batch_in_worker", None) is False:
        writer: BatchWriter
        items = writer.batch_by(items, writer.batch_size)
        # get the 1st element (2nd is the count)
        items = (writer.handle_batch(b)[0] for b in items)

    if writer is not None:
        items = writer.consolidate(items)

    return items


glob_workers = []


def adjust_num_workers(
    lc: LazyCollection,
):
    num_gpu_workers = (
        lc.num_gpu_workers
        if lc.num_gpu_workers is not None or lc.gpu_worker_devices is None
        else len(lc.gpu_worker_devices)
    )
    has_torch_pipes = any(lc.torch_components())
    requires_gpu_workers = has_torch_pipes and (
        num_gpu_workers is None or num_gpu_workers is not None and num_gpu_workers > 0
    )
    num_cpus = int(os.environ.get("EDSNLP_MAX_CPU_WORKERS") or cpu_count())
    num_devices = 0
    if requires_gpu_workers:
        import torch

        num_devices = torch.cuda.device_count()
        logging.debug(f"Number of available devices: {num_devices}")

        if num_gpu_workers is None:
            num_gpu_workers = min(num_devices, num_cpus // 2)
    else:
        num_gpu_workers = 0

    max_cpu_workers = max(num_cpus - num_gpu_workers - 1, 0)
    num_cpu_workers = (
        max_cpu_workers
        if lc.num_cpu_workers is None
        else max_cpu_workers + lc.num_cpu_workers + 1
        if lc.num_cpu_workers < 0
        else lc.num_cpu_workers
    )

    gpu_worker_devices = (
        (
            [
                f"cuda:{gpu_idx * num_devices // num_gpu_workers}"
                for gpu_idx in range(num_gpu_workers)
            ]
            if lc.gpu_worker_devices is None
            else lc.gpu_worker_devices
        )
        if requires_gpu_workers
        else []
    )
    cpu_worker_devices = (
        ["cpu"] * num_cpu_workers
        if lc.cpu_worker_devices is None
        else lc.cpu_worker_devices
    )
    assert len(cpu_worker_devices) == num_cpu_workers
    assert len(gpu_worker_devices) == num_gpu_workers
    if num_cpu_workers == 0:  # pragma: no cover
        (
            num_cpu_workers,
            num_gpu_workers,
            cpu_worker_devices,
            gpu_worker_devices,
        ) = (num_gpu_workers, 0, gpu_worker_devices, [])

    return (
        num_cpu_workers,
        num_gpu_workers,
        cpu_worker_devices,
        gpu_worker_devices,
        has_torch_pipes,
    )


def get_multiprocessing_context(has_torch_pipes, process_start_method):
    if has_torch_pipes:
        if process_start_method == "fork":
            warnings.warn(
                "Using fork start method with GPU workers may lead to deadlocks. "
                "Consider using process_start_method='spawn' instead."
            )

        process_start_method = process_start_method or "spawn"

    default_method = (
        "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    )
    if process_start_method is not None and default_method != process_start_method:
        logging.info(f"Switching process start method to {process_start_method}")
    process_start_method = process_start_method or default_method

    return multiprocessing.get_context(process_start_method)


def setup_environ(disable_implicit_parallelism):
    old_environ = {
        k: os.environ[k]
        for k in (
            "TOKENIZERS_PARALLELISM",
            "OMP_NUM_THREADS",
            "TORCH_SHARING_STRATEGY",
        )
        if k in os.environ
    }
    if disable_implicit_parallelism:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"

    if "torch" in sys.modules:
        try:
            import torch.multiprocessing

            os.environ["TORCH_SHARING_STRATEGY"] = (
                torch.multiprocessing.get_sharing_strategy()
            )
        except ImportError:  # pragma: no cover
            pass

    def revert():
        os.environ.update(old_environ)

    return revert
