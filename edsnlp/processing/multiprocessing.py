from __future__ import annotations

import copyreg
import gc
import io
import logging
import multiprocessing
import multiprocessing.reduction
import os
import sys
import tempfile
import warnings
from contextlib import nullcontext
from multiprocessing.connection import wait
from random import shuffle
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import dill
from typing_extensions import TypedDict

from edsnlp.core.lazy_collection import LazyCollection
from edsnlp.data.converters import set_current_tokenizer
from edsnlp.utils.collections import batchify, batchify_with_count, flatten_once

if TYPE_CHECKING:
    import torch

    from edsnlp.core.torch_component import TorchComponent

Stage = TypedDict(
    "Stage",
    {
        "cpu_components": List[Tuple[Callable, Dict]],
        "gpu_component": Optional[Any],
    },
)


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
        multiprocessing.reduction.ForkingPickler._extra_reducers
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


# Should we check if the multiprocessing module of edsnlp
# is responsible for this child process before replacing the pickler ?
if (
    multiprocessing.current_process() != "MainProcess"
    or hasattr(multiprocessing, "parent_process")
    and multiprocessing.parent_process() is not None
):
    replace_pickler()

DEBUG = False

debug = (
    (lambda *args, flush=False, **kwargs: print(*args, **kwargs, flush=True))
    if DEBUG
    else lambda *args, **kwargs: None
)

try:
    import torch

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
            return torch.save(*args, pickle_module=dill, **kwargs)
        finally:
            dill.settings["recurse"] = False
            if AlignDevicesHook is not None:
                del dill.Pickler.dispatch[AlignDevicesHook]
                if old is not None:  # pragma: no cover
                    dill.Pickler.dispatch[AlignDevicesHook] = old

    def load(*args, map_location=None, **kwargs):
        global MAP_LOCATION
        MAP_LOCATION = map_location
        if torch.__version__ >= "2.1.2":
            kwargs["mmap"] = True
        result = torch.load(
            *args,
            pickle_module=dill,
            map_location=map_location,
            **kwargs,
        )
        MAP_LOCATION = None
        return result

except ImportError:  # pragma: no cover

    def load(*args, map_location=None, **kwargs):
        return dill.load(*args, **kwargs)


class Exchanger:
    def __init__(
        self,
        mp: multiprocessing.context.BaseContext,
        num_stages: int,
        num_gpu_workers: int,
        num_cpu_workers: int,
        gpu_worker_devices: List[Any],
    ):
        # queue for cpu input tasks
        self.gpu_worker_devices = gpu_worker_devices
        # We add prioritized queue at the end for STOP signals
        self.cpu_inputs_queues = [
            [mp.Queue()] + [mp.SimpleQueue() for _ in range(num_stages + 1)]
            # The input queue is not shared between processes, since calling `wait`
            # on a queue reader from multiple processes may lead to a deadlock
            for _ in range(num_cpu_workers)
        ]
        self.gpu_inputs_queues = [
            [mp.Queue() for _ in range(num_stages + 1)] for _ in range(num_gpu_workers)
        ]
        self.outputs_queue = mp.Queue()

    def get_cpu_tasks(self, idx):
        while True:
            queue_readers = wait(
                [queue._reader for queue in self.cpu_inputs_queues[idx]]
            )
            stage, queue = next(
                (stage, q)
                for stage, q in reversed(list(enumerate(self.cpu_inputs_queues[idx])))
                if q._reader in queue_readers
            )
            try:
                item = queue.get()
            except BaseException:
                # Shouldn't we quit instead here ? I can't remember why we continue
                continue

            # Prioritized STOP signal (something bad happened in another process)
            # -> stop listening to input queues and raise StopIteration (return)
            if item is None and stage == len(self.cpu_inputs_queues[idx]) - 1:
                return
            yield stage, item

    def put_cpu(self, item, stage, idx):
        return self.cpu_inputs_queues[idx][stage].put(item)

    def get_gpu_tasks(self, idx):
        while True:
            queue_readers = wait(
                [queue._reader for queue in self.gpu_inputs_queues[idx]]
            )
            stage, queue = next(
                (stage, q)
                for stage, q in reversed(list(enumerate(self.gpu_inputs_queues[idx])))
                if q._reader in queue_readers
            )
            try:
                item = queue.get()
            except BaseException:  # pragma: no cover
                continue
            if item is None:
                return
            yield stage, item

    def put_gpu(self, item, stage, idx):
        return self.gpu_inputs_queues[idx][stage].put(item)

    def put_results(self, items):
        self.outputs_queue.put(items)

    def iter_results(self):
        for out in iter(self.outputs_queue.get, None):
            yield out


class CPUWorker:
    def __init__(
        self,
        cpu_idx: int,
        exchanger: Exchanger,
        gpu_pipe_names: List[str],
        lazy_collection_path: str,
        device: Union[str, "torch.device"],
    ):
        super(CPUWorker, self).__init__()

        self.cpu_idx = cpu_idx
        self.exchanger = exchanger
        self.gpu_pipe_names = gpu_pipe_names
        self.lazy_collection_path = lazy_collection_path
        self.device = device

    def _run(self):
        # Cannot pass torch tensor during init i think ? otherwise i get
        # ValueError: bad value(s) in fds_to_keep
        # mp._prctl_pr_set_pdeathsig(signal.SIGINT)

        had_error = False
        expect_new_tasks = True

        def read_tasks():
            nonlocal next_batch_id, expect_new_tasks, had_error
            iterator = self.exchanger.get_cpu_tasks(self.cpu_idx)
            while expect_new_tasks or len(active_batches) > 0:
                try:
                    stage, task = next(iterator)
                except StopIteration:
                    had_error = True
                    return
                # Non prioritized STOP signal: there are no more tasks to process and
                # we should smoothly stop (wait that there are no more active tasks,
                # and finalize the writer)
                if stage == 0 and task is None:
                    expect_new_tasks = False
                    continue

                # If first stage, we receive tasks that may require batching
                # again => we split them into subtasks
                if stage == 0:
                    task_id, fragments = task
                    for subtask in batchify(
                        lc.reader.read_worker(fragments), lc.batch_size
                    ):
                        batch_id = next_batch_id
                        active_batches[batch_id] = (subtask, task_id)
                        next_batch_id += 1
                        # gpu_idx = None
                        # batch_id : we have just created a new batch
                        # result from the last stage: None
                        yield stage, (None, batch_id, None)
                else:
                    yield stage, task

        with open(self.lazy_collection_path, "rb") as f:
            lc = load(f, map_location=self.device)
        # for name, pipe, *rest in lc.pipeline:
        #    move_to_device(pipe, self.device)

        stages: List[Stage] = [{"cpu_components": [], "gpu_component": None}]
        for name, pipe, *rest in lc.pipeline:
            if name in self.gpu_pipe_names:
                stages[-1]["gpu_component"] = pipe
                stages.append({"cpu_components": [], "gpu_component": None})
            else:
                stages[-1]["cpu_components"].append((pipe, *rest))

        next_batch_id = 0
        active_batches = {}

        logging.info(f"Starting cpu {self.cpu_idx}, PID {os.getpid()}")
        self.exchanger.outputs_queue.put(None)
        for stage, (gpu_idx, batch_id, result) in read_tasks():
            if had_error:
                continue  # pragma: no cover
            try:
                docs, task_id = active_batches.pop(batch_id)
                if stage > 0:
                    gpu_pipe = stages[stage - 1]["gpu_component"]
                    docs = gpu_pipe.postprocess(docs, result)  # type: ignore

                for pipe, kwargs, tokenizer in stages[stage]["cpu_components"]:
                    with set_current_tokenizer(tokenizer):
                        if hasattr(pipe, "batch_process"):
                            docs = pipe.batch_process(docs)
                        else:
                            docs = [pipe(doc, **kwargs) for doc in docs]

                gpu_pipe: "TorchComponent" = stages[stage]["gpu_component"]
                if gpu_pipe is not None:
                    preprocessed = gpu_pipe.make_batch(docs)  # type: ignore
                    active_batches[batch_id] = (docs, task_id)
                    if gpu_idx is None:
                        gpu_idx = batch_id % len(self.exchanger.gpu_worker_devices)
                    collated = gpu_pipe.collate(preprocessed)
                    collated = gpu_pipe.batch_to_device(
                        collated,
                        device=self.exchanger.gpu_worker_devices[gpu_idx],
                    )
                    self.exchanger.put_gpu(
                        item=(self.cpu_idx, batch_id, collated),
                        idx=gpu_idx,
                        stage=stage,
                    )
                    batch_id += 1
                else:
                    results, count = (
                        lc.writer.write_worker(docs)
                        if lc.writer is not None
                        else (docs, len(docs))
                    )
                    self.exchanger.put_results(
                        (
                            results,
                            count,
                            self.cpu_idx,
                            task_id,
                        )
                    )
            except BaseException as e:
                had_error = True
                import traceback

                print(traceback.format_exc(), flush=True)
                self.exchanger.put_results((e, 0, self.cpu_idx, None))

        if not had_error:
            if lc.writer is not None:
                results, count = lc.writer.finalize()
                if count > 0:
                    self.exchanger.put_results((results, count, self.cpu_idx, None))
        # We need to drain the queues of GPUWorker fed inputs (pre-moved to GPU)
        # to ensure no tensor allocated on producer processes (CPUWorker via
        # collate) are left in consumer processes
        [None for _ in self.exchanger.get_cpu_tasks(self.cpu_idx)]

    def run(self):
        self._run()
        gc.collect()
        sys.modules["torch"].cuda.empty_cache()


class GPUWorker:
    def __init__(
        self,
        gpu_idx,
        exchanger: Exchanger,
        gpu_pipe_names: List[str],
        lazy_collection_path: str,
        device: Union[str, "torch.device"],
    ):
        super().__init__()

        self.device = device
        self.gpu_idx = gpu_idx
        self.exchanger = exchanger

        self.gpu_pipe_names = gpu_pipe_names
        self.lazy_collection_path = lazy_collection_path
        self.device = device

    def _run(self):
        import torch

        # mp._prctl_pr_set_pdeathsig(signal.SIGINT)
        had_error = False

        with open(self.lazy_collection_path, "rb") as f:
            lc = load(f, map_location=self.device)
        stage_components = [
            pipe
            # move_to_device(pipe, self.device)
            for name, pipe, *_ in lc.pipeline
            if name in self.gpu_pipe_names
        ]
        del lc
        logging.info(f"Starting gpu {self.gpu_idx}")
        self.exchanger.outputs_queue.put(None)
        with torch.no_grad():
            for stage, task in self.exchanger.get_gpu_tasks(self.gpu_idx):
                if had_error:
                    continue  # pragma: no cover
                try:
                    cpu_idx, batch_id, batch = task
                    component = stage_components[stage]
                    res = component.module_forward(batch)
                    del batch, task
                    # TODO set non_blocking=True here
                    # res = {
                    #     key: val.to("cpu") if hasattr(val, "to") else val
                    #     for key, val in res.items()
                    # }
                    self.exchanger.put_cpu(
                        item=(self.gpu_idx, batch_id, res),
                        stage=stage + 1,
                        idx=cpu_idx,
                    )
                except BaseException as e:
                    had_error = True
                    self.exchanger.put_results((e, 0, None, None))
                    import traceback

                    print(traceback.format_exc(), flush=True)
                task = batch = res = None  # noqa
            # We need to drain the queues of CPUWorker fed inputs (pre-moved to GPU)
            # to ensure no tensor allocated on producer processes (CPUWorker via
            # collate) are left in consumer processes
            [None for _ in self.exchanger.get_gpu_tasks(self.gpu_idx)]

    def run(self):
        self._run()
        gc.collect()
        sys.modules["torch"].cuda.empty_cache()


DEFAULT_MAX_CPU_WORKERS = 4


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

    """
    try:
        TorchComponent = sys.modules["edsnlp.core.torch_component"].TorchComponent
    except (KeyError, AttributeError):  # pragma: no cover
        TorchComponent = None

    steps = lc.pipeline
    num_cpu_workers = lc.num_cpu_workers
    num_gpu_workers = lc.num_gpu_workers
    show_progress = lc.show_progress
    process_start_method = lc.process_start_method

    # Infer which pipes should be accelerated on GPU
    gpu_steps_candidates = (
        [name for name, component, *_ in steps if isinstance(component, TorchComponent)]
        if TorchComponent is not None
        else []
    )
    gpu_pipe_names = (
        gpu_steps_candidates if lc.gpu_pipe_names is None else lc.gpu_pipe_names
    )
    if set(gpu_pipe_names) - set(gpu_steps_candidates):
        raise ValueError(
            "GPU accelerated pipes {} could not be found in the model".format(
                sorted(set(gpu_pipe_names) - set(gpu_steps_candidates))
            )
        )

    requires_gpu = (
        num_gpu_workers is None
        and len(gpu_pipe_names)
        or num_gpu_workers is not None
        and num_gpu_workers > 0
    )

    num_devices = 0
    if requires_gpu:
        import torch

        num_devices = torch.cuda.device_count()
        logging.info(f"Number of available devices: {num_devices}")

        if num_gpu_workers is None:
            num_gpu_workers = num_devices

        if num_gpu_workers and process_start_method == "fork":
            warnings.warn(
                "Using fork start method with GPU workers may lead to deadlocks. "
                "Consider using process_start_method='spawn' instead."
            )

        process_start_method = process_start_method or "spawn"
    else:
        num_gpu_workers = 0

    default_method = multiprocessing.get_start_method()
    if process_start_method is not None and default_method != process_start_method:
        logging.info(f"Switching process start method to {process_start_method}")

    mp = multiprocessing.get_context(process_start_method)
    max_workers = max(min(mp.cpu_count() - num_gpu_workers, DEFAULT_MAX_CPU_WORKERS), 0)
    num_cpu_workers = (
        (num_gpu_workers or max_workers)
        if num_cpu_workers is None
        else max_workers + num_cpu_workers + 1
        if num_cpu_workers < 0
        else num_cpu_workers
    )

    if num_gpu_workers == 0:
        gpu_pipe_names = []

    gpu_worker_devices = (
        [
            f"cuda:{gpu_idx * num_devices // num_gpu_workers}"
            for gpu_idx in range(num_gpu_workers)
        ]
        if requires_gpu and lc.gpu_worker_devices is None
        else []
        if lc.gpu_worker_devices is None
        else lc.gpu_worker_devices
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
            gpu_pipe_names,
        ) = (num_gpu_workers, 0, gpu_worker_devices, [], [])

    exchanger = Exchanger(
        mp,
        num_stages=len(gpu_pipe_names),
        num_cpu_workers=num_cpu_workers,
        num_gpu_workers=num_gpu_workers,
        gpu_worker_devices=gpu_worker_devices,
    )

    lc = lc.to("cpu")

    cpu_workers = []
    gpu_workers = []

    with tempfile.NamedTemporaryFile(delete=False) as fp:
        dump(lc.worker_copy(), fp)
        fp.close()

    revert_pickler = replace_pickler()

    for gpu_idx in range(num_gpu_workers):
        gpu_workers.append(
            mp.Process(
                target=GPUWorker(
                    gpu_idx=gpu_idx,
                    exchanger=exchanger,
                    gpu_pipe_names=gpu_pipe_names,
                    lazy_collection_path=fp.name,
                    device=gpu_worker_devices[gpu_idx],
                ).run
            )
        )

    for cpu_idx in range(num_cpu_workers):
        cpu_workers.append(
            mp.Process(
                target=CPUWorker(
                    cpu_idx=cpu_idx,
                    exchanger=exchanger,
                    gpu_pipe_names=gpu_pipe_names,
                    lazy_collection_path=fp.name,
                    device=cpu_worker_devices[cpu_idx],
                ).run
            )
        )

    logging.info(f"Main PID {os.getpid()}")

    logging.info(
        f"Starting {num_cpu_workers} cpu workers and {num_gpu_workers} gpu workers on "
        f"{gpu_worker_devices}, with accelerated pipes: {gpu_pipe_names}",
    )

    for worker in (*cpu_workers, *gpu_workers):
        worker.start()

    for i in range(len((*cpu_workers, *gpu_workers))):
        assert exchanger.outputs_queue.get() is None

    os.unlink(fp.name)

    logging.info("Workers are ready")

    def process():
        try:
            num_max_enqueued = 4
            # Number of input/output batch per process
            outputs_iterator = exchanger.iter_results()

            cpu_worker_indices = list(range(num_cpu_workers))
            inputs_iterator = lc.reader.read_main()
            workloads = [{} for _ in cpu_worker_indices]

            bar = nullcontext()
            if show_progress:
                from tqdm import tqdm

                bar = tqdm(smoothing=0.1)

            with bar:
                for input_task_id, (batch, batch_size) in enumerate(
                    batchify_with_count(inputs_iterator, lc.batch_size)
                ):
                    if all(
                        sum(wl.values()) >= lc.batch_size * num_max_enqueued
                        for wl in workloads
                    ):
                        outputs, count, cpu_idx, output_task_id = next(outputs_iterator)
                        if isinstance(outputs, BaseException):
                            raise outputs
                        if show_progress:
                            bar.update(count)
                        yield outputs
                        if output_task_id is not None:
                            workloads[cpu_idx].pop(output_task_id, None)

                    # Shuffle to ensure the first process does not receive all the
                    # documents in case of workload equality
                    shuffle(cpu_worker_indices)
                    cpu_idx = min(
                        cpu_worker_indices,
                        key=lambda i: sum(workloads[i].values()),
                    )
                    exchanger.put_cpu((input_task_id, batch), stage=0, idx=cpu_idx)
                    workloads[cpu_idx][input_task_id] = batch_size

                # Inform the CPU workers that there are no more tasks to process
                for i, worker in enumerate(cpu_workers):
                    exchanger.cpu_inputs_queues[i][0].put(None)

                while any(workloads):
                    outputs, count, cpu_idx, output_task_id = next(outputs_iterator)
                    if isinstance(outputs, BaseException):
                        raise outputs  # pragma: no cover
                    if show_progress:
                        bar.update(count)
                    yield outputs
                    workloads[cpu_idx].pop(output_task_id, None)
        finally:
            revert_pickler()

            # Send gpu and cpu process the order to stop processing data
            # We use the prioritized queue to ensure the stop signal is processed
            # before the next batch of data
            for i, worker in enumerate(gpu_workers):
                exchanger.gpu_inputs_queues[i][-1].put(None)
            for i, worker in enumerate(cpu_workers):
                exchanger.cpu_inputs_queues[i][-1].put(None)

            # Enqueue a final non prioritized STOP signal to ensure there remains no
            # data in the queues (cf drain loop in CPUWorker / GPUWorker)
            for i, worker in enumerate(gpu_workers):
                exchanger.gpu_inputs_queues[i][0].put(None)
            for i, worker in enumerate(gpu_workers):
                worker.join(timeout=5)
            for i, worker in enumerate(cpu_workers):
                exchanger.cpu_inputs_queues[i][0].put(None)
            for i, worker in enumerate(cpu_workers):
                worker.join(timeout=1)

            # If a worker is still alive, kill it
            # This should not happen, but for a reason I cannot explain, it does in
            # some CPU workers sometimes when we catch an error, even though each run
            # method of the workers completes cleanly. Maybe this has something to do
            # with the cleanup of these processes ?
            for i, worker in enumerate(gpu_workers):  # pragma: no cover
                if worker.is_alive():
                    logging.error(f"Killing gpu worker {i}")
                    worker.kill()
            for i, worker in enumerate(cpu_workers):  # pragma: no cover
                if worker.is_alive():
                    logging.error(f"Killing cpu worker {i}")
                    worker.kill()

    gen = process()
    return lc.writer.write_main(gen) if lc.writer is not None else flatten_once(gen)
