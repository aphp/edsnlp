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
import time
import traceback
import warnings
import weakref
from collections import defaultdict
from contextlib import nullcontext
from itertools import tee
from multiprocessing.connection import wait
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Sequence,
    TypeVar,
    Union,
)

import dill
from tqdm import tqdm

from edsnlp.core.stream import Stage, Stream, StreamSentinel
from edsnlp.data.base import BatchWriter
from edsnlp.reducers import pickler_dont_save_module_dict
from edsnlp.utils.collections import (
    batch_compress_dict,
    decompress_dict,
)

doc_size_fns = {
    "words": len,
}

if TYPE_CHECKING:
    import torch


# Singleton is important since the STOP objects may be passed to
# other processes, i.e. pickled, unpickled, while they should
# always be the same object.


class StopType:
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
        kwargs = dict(recurse=True, byref=True, **kwargs)
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


try:
    import torch.multiprocessing

    from edsnlp.utils.torch import dump, load

except (ImportError, AttributeError):  # pragma: no cover
    # noinspection PyUnusedLocal
    def load(file, *args, map_location=None, **kwargs):
        # check if path
        if isinstance(file, str):
            with open(file, "rb") as f:
                return dill.load(f, *args, **kwargs)
        return dill.load(file, *args, **kwargs)

    def dump(obj, file, skip_tensors=False, *args, **kwargs):
        return dill.dump(obj, file, *args, **kwargs)


if os.environ.get("TORCH_SHARING_STRATEGY"):  # pragma: no cover
    try:
        torch.multiprocessing.set_sharing_strategy(os.environ["TORCH_SHARING_STRATEGY"])
    except NameError:
        pass

U = TypeVar("U")
T = TypeVar("T")


def get_dispatch_schedule(
    producer_idx: U,
    producers: Sequence[U],
    consumers: Sequence[T],
) -> List[T]:
    """
    To which consumer should a given worker/producer dispatch its data to. This
    function returns a list of consumers over a period to be determined by the
    function.

    This is actually a fun problem, because we want:
    - to distribute the data evenly between consumers
    - minimize the number of distinct unique producers sending data to the consumer
      (because we move the tensors to the GPU inside the producer, which
       creates takes a bit of VRAM for each producer)

    Parameters
    ----------
    producer_idx: U
        Index of the CPU worker
    producers: Sequence[U]
        Producers, ie workers
    consumers: Sequence[T]
        Consumers, ie devices

    Returns
    -------
    List[T]
    """
    if not (isinstance(consumers, range) and isinstance(producers, range)):
        p_range = range(len(producers))
        c_range = range(len(consumers))
        schedule = get_dispatch_schedule(
            producers.index(producer_idx), p_range, c_range
        )
        return [consumers[i] for i in schedule]

    base = []

    if consumers:
        if len(consumers) < len(producers):
            r = len(producers) % len(
                consumers
            )  # number of CPU without an easy to assign GPU
            R = producers[len(producers) - r :]
            if producer_idx not in R:
                idx = producers.index(producer_idx)
                base = [consumers[idx % len(consumers)]]
                base = base * (len(consumers) // len(base))
            else:
                base = get_dispatch_schedule(producer_idx, R, consumers)
        else:
            idx = producers.index(producer_idx)
            r = len(consumers) % len(producers)
            d = len(consumers) // len(producers)
            base = list(consumers[idx * d : (idx + 1) * d])
            base = base * len(producers)
            R = consumers[-r:]
            if r > 0:
                base = base + get_dispatch_schedule(producer_idx, producers, R)
    return base


class StopSignal(BaseException):
    pass


class Worker:
    def __init__(
        self,
        uid: str,
        num_cpu_workers: int,
        data_queues: Dict[str, multiprocessing.Queue],
        main_control_queue: multiprocessing.Queue,
        worker_control_queue: multiprocessing.Queue,
        final_barrier: multiprocessing.Barrier,
        stream_path: str,
        devices: Dict[str, Union[str, "torch.device"]],
        gpu_semaphores: Dict[str, multiprocessing.Semaphore],
    ):
        super(Worker, self).__init__()

        self.uid = uid
        self.num_cpu_workers = num_cpu_workers
        self.stream_path = stream_path
        self.stream: Stream = None  # will be assigned in run
        self.stages: List[Stage] = None  # will be assigned in run
        self.devices = devices
        self.num_producers_alive = {}
        self.stop = False

        # Synchronization primitives
        self.data_queues = data_queues
        self.gpu_semaphores = gpu_semaphores
        self.main_control_queue = main_control_queue
        self.worker_control_queue = worker_control_queue
        self.final_barrier = final_barrier
        self.waiting_times = defaultdict(float)

    def on_stop(self):
        pass

    @property
    def stages_to_run(self):
        raise NotImplementedError()

    def consumer_queues(self, stage):
        raise NotImplementedError()

    def process_items(self, stage):
        raise NotImplementedError()

    def iter_tasks(self, stage, stop_mode=False):
        raise NotImplementedError()

    def run_stage_thread(self, stage):
        self.num_producers_alive[stage] = len(
            [
                name
                for name in self.data_queues
                if name.endswith(f"to-stage-{stage}_of-{self.uid}")
            ]
        )
        try:
            self.process_items(stage)
        except StopSignal:  # pragma: no cover
            if not self.stop:
                self.stop = True
                self.on_stop()
        except BaseException as e:
            if self.stop:  # pragma: no cover
                return
            print(f"Error in {self.uid}:\n{traceback.format_exc()}", flush=True)
            self.main_control_queue.put(e)
        finally:
            try:
                for _ in self.iter_tasks(stage=stage, stop_mode=True):
                    pass
            except StopSignal:
                pass
            for name, queue in self.consumer_queues(stage):
                queue.put(STOP)

    def run(self):
        threads = []
        try:
            device = self.devices.get(self.uid, "cpu")
            self.stream, self.stages = load(self.stream_path, map_location=device)
            self.stream.eval()
            stages_to_run = self.stages_to_run

            for stage_idx in stages_to_run:
                thread = threading.Thread(
                    target=self.run_stage_thread,
                    args=(stage_idx,),
                    daemon=True,
                    name=f"Worker-{self.uid}_stage-{stage_idx}",
                )
                threads.append(thread)
                thread.start()

            # Inform the main process that we are ready
            self.main_control_queue.put("READY")

            while not self.stop:
                notification = self.worker_control_queue.get()
                if notification is STOP and not self.stop:
                    self.stop = True
                    self.on_stop()

        except BaseException as e:  # pragma: no cover
            if self.stop:
                return
            self.stop = True
            self.on_stop()

            # KeyboardInterrupt occurs in every process, so we don't want to print
            # it everywhere since it will be printed in the main process
            if not isinstance(e, KeyboardInterrupt):
                print(f"Error in {self.uid}:\n{traceback.format_exc()}", flush=True)
            self.main_control_queue.put(e)
        finally:
            # print(f"Waiting time for {self.uid}", flush=True)
            # print(
            #     "\n"
            #     + "\n".join(
            #         f"Waiting time for {self.uid}/{k}: {v:.2f}"
            #         for k, v in self.waiting_times.items()
            #     )
            #     + "\n",
            #     flush=True,
            # )
            for thread in threads:
                thread.join()
            for stage in self.stages_to_run:
                for name, queue in self.consumer_queues(stage):
                    queue.close()
                    if hasattr(queue, "join_thread"):
                        queue.join_thread()

            self.final_barrier.wait()


class CPUWorker(Worker):
    def __init__(self, *args, schedule, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_batches = {}
        self.batch_devices = {}
        self.schedule = schedule
        self.stage_schedule = defaultdict(lambda: list(schedule))

    def on_stop(self):
        # Callback when stopping
        for name, sem in self.gpu_semaphores.items():
            # just unlock everything at this point
            # Ideally we shouldn't need this and let the threads finish
            # normally, but since we don't send data anymore to GPU when
            # stop is True, logically we won't receive the results from
            # the GPU and if the semaphore was just acquired before stop
            # was set to True, another thread can be stuck waiting for it
            # without any chance for the semaphore to be released.
            # So we just unlock everything here.
            sem.release()

    def process_items(self, stage):
        items = self.iter_tasks(stage)

        # If we have a torch pipe in the previous stage, we may need to postprocess the
        # results before applying subsequent ops
        if stage > 0:
            items = self.postprocess_after_forward(items, stage)

        # Apply the ops
        for op in self.stages[stage].cpu_ops:
            items = op(items)

        # If we have a torch pipe in the current stage, we need to preprocess the items
        # before sending them to the GPU workers and apply the forward pass
        if self.stages[stage].gpu_op is not None:
            self.preprocess_before_forward(items, stage)
        else:
            self.send_results(items)

    def iter_tasks(self, stage, stop_mode=False):
        if self.stream.reader.read_in_worker and stage == 0:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            worker_idx = int(self.uid[3:]) + self.num_cpu_workers * local_rank
            pool_size = self.num_cpu_workers * world_size

            if stop_mode:
                return
            task_idx = 0
            for item in iter(self.stream.reader.read_records()):
                if self.stop:  # pragma: no cover
                    raise StopSignal()
                if isinstance(item, StreamSentinel):
                    yield item
                    continue
                if task_idx % pool_size == worker_idx:
                    for item in self.stream.reader.extract_task(item):
                        if self.stop:  # pragma: no cover
                            raise StopSignal()
                        yield item

                task_idx += 1
        else:
            task_idx = -1
            deterministic = self.stream.deterministic
            schedule = self.stage_schedule[stage]

            while self.num_producers_alive[stage] > 0:
                if self.stop and not stop_mode:  # pragma: no cover
                    raise StopSignal()

                if stage > 0:
                    task_idx = (task_idx + 1) % len(schedule)
                    while schedule[task_idx] is None:
                        task_idx = (task_idx + 1) % len(schedule)

                    if deterministic:
                        prod = schedule[task_idx]
                    else:
                        rolled = schedule[task_idx:] + schedule[:task_idx]
                        producers = list(
                            dict.fromkeys(p for p in rolled if p is not None)
                        )
                        conns = [
                            self.data_queues[
                                f"from-{prod}_to-stage-{stage}_of-{self.uid}"
                            ]._reader
                            for prod in producers
                        ]
                        ready = wait(conns)[0]
                        prod = producers[conns.index(ready)]
                else:
                    prod = "main"

                name = f"from-{prod}_to-stage-{stage}_of-{self.uid}"
                queue = self.data_queues[name]
                t = time.time()
                item = queue.get()

                self.waiting_times["get-" + name] += time.time() - t
                if item is STOP:
                    schedule[:] = [s if s != prod else None for s in schedule]
                    self.num_producers_alive[stage] -= 1
                    continue
                yield item

    def postprocess_after_forward(self, items, stage):
        postprocess = getattr(self.stages[stage - 1].gpu_op, "postprocess", None)
        deterministic = self.stream.deterministic
        for batch_id, result in items:
            x = self.active_batches[stage - 1]
            slot = next(s for s in x if s[0] == batch_id)
            result = {
                k: v.to("cpu") if hasattr(v, "to") else v for k, v in result.items()
            }
            _, docs, inputs, _ = slot
            if postprocess:
                docs = postprocess(docs, result, inputs)
            else:
                docs = result
            slot[3] = docs
            if deterministic:
                while x and x[0][3] is not None:
                    docs = x.pop(0)[3]
                    yield docs
            else:
                x.remove(slot)
                yield docs

            del docs, result, inputs, slot, x

    def preprocess_before_forward(self, items, stage):
        if stage not in self.active_batches:
            self.active_batches[stage] = []
        task_idx = 0
        torch_pipe = self.stages[stage].gpu_op
        for docs in items:
            if isinstance(docs, StreamSentinel):
                self.active_batches[stage].append([None, None, None, docs])
                continue
            batch_id = str(hash(tuple(id(x) for x in docs)))[-8:] + "-" + self.uid
            torch_pipe.enable_cache(batch_id)
            if stage == 0:
                gpu_idx = self.schedule[task_idx % len(self.schedule)]
                self.batch_devices[batch_id] = gpu_idx
            else:
                # Ensure we send the batch to the same device as the previous stages.
                # In deterministic mode, using schedule[task_idx % len(schedule)]
                # we shouldn't need this, but in non-deterministic mode, the order
                # in which batches are returned to cpu workers may change depending
                # on the speed at which gpu workers process them, rendering
                # the above gpu_idx computation inconsistent over time.
                gpu_idx = self.batch_devices[batch_id]
            task_idx += 1
            device = self.devices[gpu_idx]
            if hasattr(torch_pipe, "preprocess"):
                inputs = [torch_pipe.preprocess(x) for x in docs]
                batch = decompress_dict(list(batch_compress_dict(inputs)))
                batch = torch_pipe.collate(batch)
                batch = torch_pipe.batch_to_device(batch, device=device)
            else:
                batch = torch_pipe.prepare_batch(docs, device=device)
                inputs = None

            self.active_batches[stage].append([batch_id, docs, inputs, None])

            name = f"from-{self.uid}_to-stage-{stage}_of-{gpu_idx}"
            queue = self.data_queues[name]
            item = (self.uid, batch_id, batch)
            if not self.stop and stage == 0:
                sem = self.gpu_semaphores[gpu_idx]
                t = time.time()
                sem.acquire()

                self.waiting_times[f"semaphore-{gpu_idx}"] += time.time() - t
            t = time.time()
            queue.put(item)

            self.waiting_times["put-" + name] += time.time() - t

            del batch, inputs

            if stage == len(self.stages) - 2:
                torch_pipe.disable_cache(batch_id)
                self.batch_devices.pop(batch_id, None)

    def send_results(self, items):
        writer = self.stream.writer
        if writer is not None:
            items = (
                writer.handle_record(rec)
                if not isinstance(rec, StreamSentinel)
                else rec
                for rec in items
            )
        if getattr(writer, "write_in_worker", None) is True:
            items = writer.batch_fn(
                items,
                batch_size=writer.batch_size,
                sentinel_mode="drop",
            )
            items = (
                writer.handle_batch(b)
                for b in items
                if not isinstance(b, StreamSentinel)
            )
        else:
            items = (
                (x, 1) if not isinstance(x, StreamSentinel) else (x, 0) for x in items
            )

        name = f"from-{self.uid}_to-main"
        queue = self.data_queues[name]
        for item in items:
            t = time.time()
            queue.put(item)

            self.waiting_times["put-" + name] += time.time() - t

    @property
    def stages_to_run(self):
        return range(len(self.stages))

    def consumer_queues(self, stage):
        if stage < len(self.stages) - 1:
            names = [
                f"from-{self.uid}_to-stage-{stage}_of-{gpu}"
                for gpu in set(self.schedule)
            ]
            return [(name, self.data_queues[name]) for name in names]
        else:
            name = f"from-{self.uid}_to-main"
            return [(name, self.data_queues[name])]


class GPUWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    def process_items(self, stage):
        autocast = self.stream.autocast
        autocast_ctx = nullcontext()
        device = self.devices[self.uid]
        device_type = getattr(device, "type", device).split(":")[0]
        try:
            if autocast:
                autocast_ctx = torch.autocast(
                    device_type=device_type,
                    dtype=autocast if autocast is not True else None,
                )
        except RuntimeError:  # pragma: no cover
            pass

        with torch.no_grad(), autocast_ctx, torch.inference_mode():
            for item in self.iter_tasks(stage):
                with self.lock:
                    cpu_id, batch_id, batch = item

                    pipe = self.stages[stage].gpu_op
                    pipe.enable_cache(batch_id)
                    batch = pipe(batch)

                    # Put item into the queue
                    name = f"from-{self.uid}_to-stage-{stage + 1}_of-{cpu_id}"
                    queue = self.data_queues[name]
                    item = (batch_id, batch)

                # Do NOT put during lock, otherwise this may lead to a deadlock
                # in multi-stage (n + 1 where n > 1) scenarios where stage 1
                # must put in a queue that is full, and could only be emptied
                # if subsequent stages run, but forward in stage 2 cannot run
                # since we have acquired the lock.

                if stage == len(self.stages) - 2:  # Last GPU stage
                    pipe.disable_cache(batch_id)
                    self.gpu_semaphores[cpu_id].release()
                t = time.time()
                queue.put(item)

                self.waiting_times["put-" + name] += time.time() - t

                del batch, item

    def iter_tasks(self, stage, stop_mode=False):
        offset = -1
        queues = [
            key
            for key, q in self.data_queues.items()
            if key.endswith(f"-{stage}_of-{self.uid}")
        ]
        # Get items from the previous stage
        while self.num_producers_alive[stage] > 0:
            if self.stop and not stop_mode:  # pragma: no cover
                raise StopSignal()

            offset = (offset + 1) % len(queues)
            while queues[offset] is None:  # pragma: no cover
                offset = (offset + 1) % len(queues)

            # Roll the queues to ensure we don't always get from the same queue
            names = [q for q in queues[offset:] + queues[:offset] if q is not None]
            conns = [self.data_queues[name]._reader for name in names]
            ready = wait(conns)[0]
            name = names[conns.index(ready)]
            queue = self.data_queues[name]
            t = time.time()
            item = queue.get()

            self.waiting_times["get-" + name] += time.time() - t

            if item is STOP:
                queues[:] = [s if s != name else None for s in queues]
                self.num_producers_alive[stage] -= 1
                continue
            yield item

    @property
    def stages_to_run(self):
        return range(len(self.stages) - 1)

    def consumer_queues(self, stage):
        names = [
            k
            for k, q in self.data_queues.items()
            if k.startswith(f"from-{self.uid}_to-stage-{stage + 1}_of-")
        ]
        return [(name, self.data_queues[name]) for name in names]


class SafeNext:
    def __init__(self, it, lock):
        self.it = it
        self.lock = lock

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def thread_safe_tee(it: Iterable[T], n: int):
    lock = threading.Lock()
    return tuple(SafeNext(iter(sub_it), lock) for sub_it in tee(it, n))


class MultiprocessingStreamExecutor:
    def __init__(self, stream):
        (
            num_cpu_workers,
            num_gpu_workers,
            cpu_worker_devices,
            gpu_worker_devices,
            has_torch_pipes,
        ) = self.adjust_num_workers(stream)
        self.stream = stream
        self.stages = stream._make_stages(split_torch_pipes=num_gpu_workers > 0)
        self.has_torch_pipes = has_torch_pipes
        mp = self.get_multiprocessing_context(
            has_torch_pipes=has_torch_pipes,
            process_start_method=stream.process_start_method,
        )

        # Queues definition
        share_queues = not stream.deterministic

        self.cpu_worker_names = [f"cpu{i}" for i in range(num_cpu_workers)]
        self.gpu_worker_names = [f"gpu{i}" for i in range(num_gpu_workers)]
        self.all_worker_names = self.cpu_worker_names + self.gpu_worker_names

        self.cpu_to_gpu_schedules = {
            cpu: get_dispatch_schedule(
                cpu,
                self.cpu_worker_names,
                self.gpu_worker_names,
            )
            for cpu in self.cpu_worker_names
        }

        self.main_control_queue = mp.Queue()
        self.worker_control_queues = {
            name: mp.Queue() for name in self.all_worker_names
        }
        self.final_barrier = mp.Barrier(num_cpu_workers + num_gpu_workers + 1)
        self.data_queues: Dict[str, multiprocessing.Queue] = {}
        self.gpu_semaphores = {}
        self.input_queue_names = []

        # Queues: for each N producers - 1 consumer situation, we don't create
        # one single shared queue but N bounded queues, one for each producer.
        # This is to avoid the situation where a single producer occupies all
        # slots, leading to congestions affecting the whole workers pool.

        # Input queues for each CPU worker
        if not self.stream.reader.read_in_worker:
            queue = mp.Queue(2 * num_cpu_workers) if share_queues else None
            for cpu in self.cpu_worker_names:
                name = f"from-main_to-stage-0_of-{cpu}"
                if not share_queues:
                    queue = mp.SimpleQueue()
                self.data_queues[name] = queue
                self.input_queue_names.append(name)

        for cpu in set(self.cpu_worker_names):
            for gpu in set(self.cpu_to_gpu_schedules[cpu]):
                # Control the number of active items for each CPU -> GPU pair
                self.gpu_semaphores[(cpu, gpu)] = mp.Semaphore(2)

                for stage in range(0, len(self.stages) - 1):
                    # Queue to send data from CPU to GPU
                    name = f"from-{cpu}_to-stage-{stage}_of-{gpu}"
                    self.data_queues[name] = mp.Queue()

                    # Answer queue from GPU to CPU
                    name = f"from-{gpu}_to-stage-{stage + 1}_of-{cpu}"
                    self.data_queues[name] = mp.Queue()

            # Final output queue for each CPU worker
            name = f"from-{cpu}_to-main"
            self.data_queues[name] = mp.Queue(2)

        self.cpu_temp_file = self.gpu_temp_file = None
        if len(self.cpu_worker_names):
            self.cpu_temp_file = tempfile.NamedTemporaryFile(delete=False)
        if len(self.gpu_worker_names):
            self.gpu_temp_file = tempfile.NamedTemporaryFile(delete=False)

        self.cpu_workers = []
        self.gpu_workers = []
        devices = {
            **(dict(zip(self.cpu_worker_names, cpu_worker_devices))),
            **(dict(zip(self.gpu_worker_names, gpu_worker_devices))),
        }
        for cpu in self.cpu_worker_names:
            worker = CPUWorker(
                uid=cpu,
                num_cpu_workers=num_cpu_workers,
                data_queues={k: q for k, q in self.data_queues.items() if cpu in k},
                main_control_queue=self.main_control_queue,
                worker_control_queue=self.worker_control_queues[cpu],
                final_barrier=self.final_barrier,
                schedule=self.cpu_to_gpu_schedules[cpu],
                stream_path=self.cpu_temp_file.name,
                devices=devices,
                gpu_semaphores={
                    gpu: sem
                    for (c, gpu), sem in self.gpu_semaphores.items()
                    if cpu == c
                },
            )
            self.cpu_workers.append(
                mp.Process(
                    target=worker.run,
                    name=f"CPUWorker-{cpu}",
                    daemon=True,
                )
            )
        for gpu in self.gpu_worker_names:
            worker = GPUWorker(
                uid=gpu,
                num_cpu_workers=num_cpu_workers,
                data_queues={k: q for k, q in self.data_queues.items() if gpu in k},
                main_control_queue=self.main_control_queue,
                worker_control_queue=self.worker_control_queues[gpu],
                final_barrier=self.final_barrier,
                stream_path=self.gpu_temp_file.name,
                devices=devices,
                gpu_semaphores={
                    cpu: sem
                    for (cpu, g), sem in self.gpu_semaphores.items()
                    if gpu == g
                },
            )
            self.gpu_workers.append(
                mp.Process(
                    target=worker.run,
                    name=f"GPUWorker-{gpu}",
                    daemon=True,
                )
            )
        self.stopped = False
        self.num_alive_workers = num_cpu_workers
        self.workers_status = [True] * num_cpu_workers
        self.current_worker_idx = -1
        self.error = None
        self.dequeue_notifications_thread = None
        self.queue_feeder_threads: List[threading.Thread] = []
        self.revert_environ = lambda: None
        self.revert_pickler = lambda: None
        self.tearing_down = False

        logging.debug(f"Main PID {os.getpid()}")
        logging.info(
            f"Running {num_cpu_workers} CPU workers and {num_gpu_workers} GPU "
            f"workers on {gpu_worker_devices} in {mp.get_start_method()} mode "
            f"to run {len(self.stages)} stage{'s' if len(self.stages) > 1 else ''}.",
        )

    def run(self):
        self.revert_pickler = replace_pickler()
        self.revert_environ = self.setup_environ(
            self.stream.disable_implicit_parallelism
        )

        stream_to_dump = self.stream.worker_copy()
        with pickler_dont_save_module_dict():
            if self.cpu_temp_file:
                # If we have GPU workers, these will be responsible for the forward pass
                # and CPU workers will only perform preprocessing and postprocessing
                # so they don't need deep learning parameters
                # TODO: should we make this a stream set_processing option ?
                keep_tensors = self.has_torch_pipes and len(self.gpu_worker_names) == 0
                with self.cpu_temp_file as fp:
                    dump(
                        (stream_to_dump, self.stages),
                        fp,
                        skip_tensors=not keep_tensors,
                    )
                    fp.close()
            if self.gpu_temp_file:
                with self.gpu_temp_file as fp:
                    dump((stream_to_dump, self.stages), fp)
                    fp.close()

        del stream_to_dump
        for worker in (*self.cpu_workers, *self.gpu_workers):
            worker.start()

        # Wait for all workers to be ready before deleting the temp file
        # and starting the main loop
        for _ in range(len(self.cpu_workers) + len(self.gpu_workers)):
            outputs = self.main_control_queue.get()
            if isinstance(outputs, BaseException):  # pragma: no cover
                raise outputs

        if self.cpu_temp_file is not None:
            os.unlink(self.cpu_temp_file.name)
        if self.gpu_temp_file is not None:
            os.unlink(self.gpu_temp_file.name)

        # Start listening for notifications from workers
        self.dequeue_notifications_thread = threading.Thread(
            target=self.dequeue_notifications,
            name="Main-Notifications",
            daemon=True,
        )
        self.dequeue_notifications_thread.start()

        # Start enqueuing inputs if needed
        if not self.stream.reader.read_in_worker:
            queues = {self.data_queues[name] for name in self.input_queue_names}
            tee_items = thread_safe_tee(self.stream.reader.read_records(), len(queues))
            for queue, items in zip(queues, tee_items):
                thread = threading.Thread(
                    target=self.feed_queue,
                    name="Main-Enqueue-Inputs",
                    daemon=True,
                    args=(queue, items),
                )
                thread.start()
                self.queue_feeder_threads.append(thread)

        # Create the main iterator
        items = self.dequeue_outputs()
        writer = self.stream.writer
        if getattr(writer, "write_in_worker", None) is False:
            writer: BatchWriter
            items = writer.batch_fn(items, writer.batch_size, sentinel_mode="drop")
            # get the 1st element (2nd is the count)
            items = (
                writer.handle_batch(b)[0]
                for b in items
                if not isinstance(b, StreamSentinel)
            )

        # If we are garbage collected, stop the execution
        weakref.finalize(items, self.teardown, garbage_collected=True)

        if writer is not None:
            items = writer.consolidate(items)

        return items

    def dequeue_outputs(self):
        try:
            bar = tqdm(
                smoothing=0.1,
                mininterval=1.0,
                disable=not self.stream.show_progress,
            )
            with bar:
                for item, count in self.iter_outputs():
                    bar.update(count)
                    yield item
        except StopSignal:
            if self.error:
                raise self.error
        except BaseException as e:
            self.error = e
            raise
        finally:
            self.teardown()

    def iter_outputs(self, stop_mode=False):
        deterministic = self.stream.deterministic
        requires_sentinel = (
            hasattr(self.stream.writer, "batch_fn")
            and getattr(self.stream.writer.batch_fn, "requires_sentinel", None)
            and not self.stream.writer.write_in_worker
        )
        missing_sentinels = len(self.cpu_worker_names) if requires_sentinel else 0
        buffer = []
        while self.num_alive_workers > 0:
            if self.stopped and not stop_mode:  # pragma: no cover
                raise StopSignal()

            self.current_worker_idx = (self.current_worker_idx + 1) % len(
                self.cpu_worker_names
            )
            while not self.workers_status[self.current_worker_idx]:
                self.current_worker_idx = (self.current_worker_idx + 1) % len(
                    self.cpu_worker_names
                )

            if deterministic:
                worker_idx = self.current_worker_idx
            else:
                workers = [
                    *range(self.current_worker_idx, len(self.cpu_worker_names)),
                    *range(self.current_worker_idx),
                ]
                workers = [idx for idx in workers if self.workers_status[idx]]
                output_conns = [
                    self.data_queues[
                        f"from-{self.cpu_worker_names[idx]}_to-main"
                    ]._reader
                    for idx in workers
                ]
                pipe = wait(output_conns)[0]
                worker_idx = workers[output_conns.index(pipe)]

            name = f"from-{self.cpu_worker_names[worker_idx]}_to-main"
            queue = self.data_queues[name]

            out = queue.get()
            if out is STOP:
                self.num_alive_workers -= 1
                self.workers_status[worker_idx] = False
                continue
            if isinstance(out[0], StreamSentinel):
                if out[0].kind == requires_sentinel:
                    missing_sentinels -= 1
                    if missing_sentinels == 0:
                        yield from buffer
                        yield out
                        buffer.clear()
                        missing_sentinels = len(self.cpu_worker_names)
                continue
            if requires_sentinel:
                buffer.append(out)
            else:
                yield out
        yield from buffer
        if self.error:
            raise self.error

    def feed_queue(self, queue, items):
        """
        Enqueue items in a queue.
        Note that a queue may be shared between multiple workers, so have to send
        items destined multiple workers in the same queue. For that, we first
        determine which worker should receive the item based on the item index
        and some other env variables. Then we lookup the worker queue, and if it
        matches the current queue, we send the item, even if all workers share the
        same queue, in which case there is only on queue feeder thread that sends
        all the items (non-deterministic mode).

        Parameters
        ----------
        queue: multiprocessing.Queue
            The queue to feed
        items: Iterator
            The items to send. Note that this iterator is a tee of the main
            iterator, such that each worker can process items at its own pace.
        """
        queues = [
            self.data_queues[f"from-main_to-stage-0_of-{cpu}"]
            for cpu in self.cpu_worker_names
        ]
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

            task_idx = 0
            for item in items:
                if self.stopped:
                    break
                if isinstance(item, StreamSentinel):
                    queue.put(item)
                    continue
                # tasks:         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ...]
                # world_size = 2
                # local_rank = 1
                # -> kept tasks: [   1,    3,    5,    7,    9, ...]
                # worker_idx = 1
                # num_cpu_workers = 3
                # -> kept tasks: [         3,                9, ...]
                if (
                    # check that this task is for us
                    (task_idx % world_size) == local_rank
                    # check that this task is for the queue
                    and queues[(task_idx // world_size) % len(queues)] is queue
                ):
                    queue.put(item)
                task_idx += 1
        except BaseException as e:
            print(f"Error in input enqueueing:\n{traceback.format_exc()}", flush=True)
            self.main_control_queue.put(e)
        finally:
            # Send the stop sentinel to all workers
            for q in queues:
                if q is queue:
                    queue.put(STOP)
            if hasattr(queue, "close"):
                queue.close()
            if hasattr(queue, "join_thread"):
                queue.join_thread()

    def dequeue_notifications(self):
        while True:
            notification = self.main_control_queue.get()
            if notification is STOP:
                break
            if isinstance(notification, BaseException):
                self.error = notification
                self.send_stop_signals()

    def teardown(self, garbage_collected=False):
        if self.tearing_down:
            return
        self.tearing_down = True

        if self.error:
            warnings.warn(
                "An error occurred. Cleaning up resources, please hang tight...",
            )
        elif garbage_collected:
            warnings.warn(
                "Multiprocessing executor was garbage collected while still running. "
                "Cleaning up resources, please hang tight...",
            )

        self.send_stop_signals()

        self.revert_environ()
        self.revert_pickler()

        for _ in self.iter_outputs(stop_mode=True):
            pass

        if self.dequeue_notifications_thread is not None:
            self.dequeue_notifications_thread.join()
        for thread in self.queue_feeder_threads:
            thread.join()
        self.final_barrier.wait()

    def send_stop_signals(self):
        if self.stopped:
            return

        self.stopped = True

        for uid in (*self.cpu_worker_names, *self.gpu_worker_names):
            self.worker_control_queues[uid].put(STOP)
        self.main_control_queue.put(STOP)

    @staticmethod
    def adjust_num_workers(stream: Stream):
        num_gpu_workers = (
            stream.num_gpu_workers
            if stream.num_gpu_workers is not None or stream.gpu_worker_devices is None
            else len(stream.gpu_worker_devices)
        )
        has_torch_pipes = any(stream.torch_components())
        requires_gpu_workers = has_torch_pipes and (
            num_gpu_workers is None
            or num_gpu_workers is not None
            and num_gpu_workers > 0
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
        default_cpu_workers = max(
            min(max_cpu_workers, num_gpu_workers * 4)
            if num_gpu_workers > 0
            else max_cpu_workers,
            1,
        )
        num_cpu_workers = (
            default_cpu_workers
            if stream.num_cpu_workers is None
            else max_cpu_workers + stream.num_cpu_workers + 1
            if stream.num_cpu_workers < 0
            else stream.num_cpu_workers
        )

        gpu_worker_devices = (
            (
                [
                    f"cuda:{gpu_idx * num_devices // num_gpu_workers}"
                    for gpu_idx in range(num_gpu_workers)
                ]
                if stream.gpu_worker_devices is None
                else stream.gpu_worker_devices
            )
            if requires_gpu_workers
            else []
        )
        cpu_worker_devices = (
            ["cpu"] * num_cpu_workers
            if stream.cpu_worker_devices is None
            else stream.cpu_worker_devices
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

        if num_cpu_workers == 0:
            raise ValueError("At least one worker is required in multiprocessing mode.")

        num_cpu_workers: int
        num_gpu_workers: int
        cpu_worker_devices: List[str]
        gpu_worker_devices: List[str]
        has_torch_pipes: bool
        return (
            num_cpu_workers,
            num_gpu_workers,
            cpu_worker_devices,
            gpu_worker_devices,
            has_torch_pipes,
        )

    @staticmethod
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

    @staticmethod
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


def execute_multiprocessing_backend(stream: Stream):
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
    """
    executor = MultiprocessingStreamExecutor(stream)
    return executor.run()
