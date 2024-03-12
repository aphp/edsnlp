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
from typing_extensions import TypedDict

from edsnlp.core.lazy_collection import LazyCollection
from edsnlp.data.converters import set_current_tokenizer
from edsnlp.utils.collections import batchify, flatten_once

batch_size_fns = {
    "words": lambda batch: sum(len(doc) for doc in batch),
    "padded_words": lambda batch: max(len(doc) for doc in batch) * len(batch),
    "docs": len,
}

doc_size_fns = {
    "words": len,
}

if TYPE_CHECKING:
    import torch

    from edsnlp.core.torch_component import TorchComponent

Stage = TypedDict(
    "Stage",
    {
        "cpu_components": List[Tuple[str, Callable, Dict, Any]],
        "gpu_component": Optional[Any],
    },
)


def apply_basic_pipes(docs, pipes):
    for name, pipe, kwargs, tok in pipes:
        with set_current_tokenizer(tok):
            if hasattr(pipe, "batch_process"):
                docs = pipe.batch_process(docs)
            else:
                docs = [pipe(doc, **kwargs) for doc in docs]
    return docs


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

DEBUG = True

debug = (
    (lambda *args, flush=False, **kwargs: print(*args, **kwargs, flush=True))
    if DEBUG
    else lambda *args, **kwargs: None
)

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
        # with open(args[0], "rb") as f:
        #     result = dill.load(f, **kwargs)
        try:
            if torch.__version__ < "2.0.0":
                pickle = torch_load.__globals__["pickle"]
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

    def load(file, *args, map_location=None, **kwargs):
        # check if path
        if isinstance(file, str):
            with open(file, "rb") as f:
                return dill.load(f, *args, **kwargs)
        return dill.load(file, *args, **kwargs)

    dump = dill.dump


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
        self.num_cpu_workers = num_cpu_workers
        self.num_gpu_workers = num_gpu_workers
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
        self.num_stages = num_stages

    # noinspection PyUnresolvedReferences
    def get_cpu_task(self, idx, get_instant_active_or_skip: bool = False):
        queues = self.cpu_inputs_queues[idx]
        if get_instant_active_or_skip:
            # Don't get new tasks
            queues = queues[1:]
        queue_readers = wait(
            [queue._reader for queue in queues],
            timeout=0 if get_instant_active_or_skip else None,
        )
        if len(queue_readers) == 0:
            return None, None
        stage, queue = next(
            (stage, q)
            for stage, q in reversed(list(enumerate(self.cpu_inputs_queues[idx])))
            if q._reader in queue_readers
        )
        item = queue.get()
        return stage, item

    def put_cpu(self, item, stage, idx):
        return self.cpu_inputs_queues[idx][stage].put(item)

    def get_gpu_task(self, idx):
        queue_readers = wait([queue._reader for queue in self.gpu_inputs_queues[idx]])
        stage, queue = next(
            (stage, q)
            for stage, q in reversed(list(enumerate(self.gpu_inputs_queues[idx])))
            if q._reader in queue_readers
        )
        item = queue.get()
        return stage, item

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

    def run(self):
        # Cannot pass torch tensor during init i think ? otherwise i get
        # ValueError: bad value(s) in fds_to_keep
        # mp._prctl_pr_set_pdeathsig(signal.SIGINT)
        next_batch_id = self.cpu_idx
        new_batch_iterator = None

        def split_task_into_new_batches(task):
            nonlocal next_batch_id, new_batch_iterator
            task_id, fragments = task
            chunks = list(batchify(lc.reader.read_worker(fragments), lc.chunk_size))
            for chunk_idx, docs in enumerate(chunks):
                # If we sort by size, we must first create the documents
                # to have features against which we will sort
                docs = apply_basic_pipes(docs, preprocess_pipes)

                if lc.sort_chunks:
                    docs.sort(key=doc_size_fns.get(lc.sort_chunks, len))

                batches = [
                    batch
                    for batch in batchify(
                        docs,
                        batch_size=lc.batch_size,
                        formula=batch_size_fns[lc.batch_by],
                    )
                ]

                for batch_idx, batch in enumerate(batches):
                    assert len(batch) > 0
                    batch_id = next_batch_id

                    # We mark the task id only for the last batch of a task
                    # since the purpose of storing the task id is to know
                    # when the worker has finished processing the task,
                    # which is true only when the last batch has been
                    # processed
                    active_batches[batch_id] = (
                        batch,
                        task_id
                        if (batch_idx == len(batches) - 1)
                        and (chunk_idx == len(chunks) - 1)
                        else None,
                    )
                    next_batch_id += num_cpu
                    # gpu_idx = None
                    # batch_id = we have just created a new batch
                    # result from the last stage = None
                    if batch_idx == len(batches) - 1 and chunk_idx == len(chunks) - 1:
                        new_batch_iterator = None
                    yield 0, (None, batch_id, None)

            new_batch_iterator = None

        def read_tasks():
            nonlocal new_batch_iterator

            expect_new_tasks = True

            while expect_new_tasks or len(active_batches) > 0:
                # Check that there are no more than `chunk_size` docs being processed.
                # If there is still room, we can process new batches
                has_room_for_new_batches = (
                    sum(len(ab[0]) for ab in active_batches.values()) < lc.chunk_size
                )

                # if new_batch_iterator is not None and len(active_batches) == 0:
                #     yield next(new_batch_iterator)
                #     continue

                stage, task = self.exchanger.get_cpu_task(
                    idx=self.cpu_idx,
                    # We don't have to wait for new active batches to come back if:
                    get_instant_active_or_skip=(
                        # - we have room for more batches
                        has_room_for_new_batches
                        # - and the batch iterator is still active
                        and new_batch_iterator is not None
                    ),
                )

                # No active batch was returned, and by construction we have room for
                # new batches, so we can start a new batch
                if stage is None:
                    yield next(new_batch_iterator)
                    continue

                # stage, task = next(iterator)
                # Prioritized STOP signal: something bad happened in another process
                # -> stop listening to input queues and raise StopIteration (return)
                if task is None and stage == self.exchanger.num_stages + 1:
                    return
                # Non prioritized STOP signal: there are no more tasks to process
                # and we should smoothly stop (wait that there are no more active
                # tasks, and finalize the writer)
                if stage == 0 and task is None:
                    expect_new_tasks = False
                    continue

                # If first stage, we receive tasks that may require batching
                # again => we split them into chunks
                if stage == 0:
                    new_batch_iterator = split_task_into_new_batches(task)
                    yield next(new_batch_iterator)
                else:
                    yield stage, task

        try:
            lc: LazyCollection = load(
                self.lazy_collection_path, map_location=self.device
            )
            lc.eval()
            preprocess_pipes = []
            num_cpu = self.exchanger.num_cpu_workers
            split_into_batches_after = lc.split_into_batches_after
            if (
                split_into_batches_after is None
                or lc.batch_by != "docs"
                or lc.sort_chunks
            ):
                split_into_batches_after = next(
                    (p[0] for p in lc.pipeline if p[0] is not None), None
                )
            is_before_split = split_into_batches_after is not None

            stages: List[Stage] = [{"cpu_components": [], "gpu_component": None}]
            for name, pipe, *rest in lc.pipeline:
                if name in self.gpu_pipe_names:
                    is_before_split = False
                    stages[-1]["gpu_component"] = pipe
                    stages.append({"cpu_components": [], "gpu_component": None})
                else:
                    if is_before_split:
                        preprocess_pipes.append((name, pipe, *rest))
                    else:
                        stages[-1]["cpu_components"].append((name, pipe, *rest))
                if name is split_into_batches_after:
                    is_before_split = False

            # Start at cpu_idx to avoid having all workers sending their
            # first batch (0 % num_device, cf below) to the same gpu
            active_batches = {}

            logging.info(f"Starting {self} on {os.getpid()}")

            # Inform the main process that we are ready
            self.exchanger.put_results((None, 0, None, None))

            for stage, (gpu_idx, batch_id, result) in read_tasks():
                docs, task_id = active_batches.pop(batch_id)
                for name, pipe, *rest in lc.pipeline:
                    if hasattr(pipe, "enable_cache"):
                        pipe.enable_cache(batch_id)
                if stage > 0:
                    gpu_pipe = stages[stage - 1]["gpu_component"]
                    docs = gpu_pipe.postprocess(docs, result)  # type: ignore

                docs = apply_basic_pipes(docs, stages[stage]["cpu_components"])

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
                else:
                    for name, pipe, *rest in lc.pipeline:
                        if hasattr(pipe, "disable_cache"):
                            pipe.disable_cache(batch_id)
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

            results, count = lc.writer.finalize() if lc.writer is not None else ([], 0)
            self.exchanger.put_results((results, count, self.cpu_idx, "finalize"))

        except BaseException as e:
            import traceback

            print(f"Error in {self}:\n{traceback.format_exc()}", flush=True)
            self.exchanger.put_results((e, 0, self.cpu_idx, None))
        # We need to drain the queues of GPUWorker fed inputs (pre-moved to GPU)
        # to ensure no tensor allocated on producer processes (CPUWorker via
        # collate) are left in consumer processes
        task = True  # anything but None
        stage = None
        while (stage, task) != (0, None):
            try:
                stage, task = self.exchanger.get_cpu_task(self.cpu_idx)
            finally:
                pass

    def __repr__(self):
        return f"<CPUWorker idx={self.cpu_idx}>"


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

    def run(self):
        import torch

        # mp._prctl_pr_set_pdeathsig(signal.SIGINT)
        try:
            lc = load(self.lazy_collection_path, map_location=self.device)
            lc.eval()
            stage_components = [
                pipe
                # move_to_device(pipe, self.device)
                for name, pipe, *_ in lc.pipeline
                if name in self.gpu_pipe_names
            ]

            del lc
            logging.info(f"Starting {self} on {os.getpid()}")

            # Inform the main process that we are ready
            self.exchanger.put_results((None, 0, None, None))

            with torch.no_grad():
                while True:
                    stage, task = self.exchanger.get_gpu_task(self.gpu_idx)
                    if task is None:
                        break

                    cpu_idx, batch_id, batch = task
                    pipe = stage_components[stage]
                    pipe.enable_cache(batch_id)
                    res = pipe.module_forward(batch)
                    self.exchanger.put_cpu(
                        item=(
                            self.gpu_idx,
                            batch_id,
                            {
                                k: v.to("cpu") if hasattr(v, "to") else v
                                for k, v in res.items()
                            },
                        ),
                        stage=stage + 1,
                        idx=cpu_idx,
                    )
                    if stage == len(stage_components) - 1:
                        pipe.disable_cache(batch_id)
                    del batch, task

                task = batch = res = None  # noqa
        except BaseException as e:
            import traceback

            print(f"Error in {self}:\n{traceback.format_exc()}", flush=True)
            self.exchanger.put_results((e, 0, None, None))

        from edsnlp.core.torch_component import _caches

        task = batch = res = None  # noqa
        _caches.clear()
        gc.collect()
        sys.modules["torch"].cuda.empty_cache()

        # We need to drain the queues of CPUWorker fed inputs (pre-moved to GPU)
        # to ensure no tensor allocated on producer processes (CPUWorker via
        # collate) are left in consumer processes
        stage = None
        task = None
        while (stage, task) != (0, None):
            try:
                stage, task = self.exchanger.get_gpu_task(self.gpu_idx)
            finally:
                pass

    def __repr__(self):
        return f"<GPUWorker idx={self.gpu_idx}>"


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

    old_environ = {
        k: os.environ.get(k) for k in ("TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS")
    }
    if lc.disable_implicit_parallelism:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"

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
    else:
        num_gpu_workers = 0

    if any(gpu_steps_candidates):
        if process_start_method == "fork":
            warnings.warn(
                "Using fork start method with GPU workers may lead to deadlocks. "
                "Consider using process_start_method='spawn' instead."
            )

        process_start_method = process_start_method or "spawn"

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
                target=GPUWorker.run,
                args=(
                    GPUWorker(
                        gpu_idx=gpu_idx,
                        exchanger=exchanger,
                        gpu_pipe_names=gpu_pipe_names,
                        lazy_collection_path=fp.name,
                        device=gpu_worker_devices[gpu_idx],
                    ),
                ),
            )
        )

    for cpu_idx in range(num_cpu_workers):
        cpu_workers.append(
            mp.Process(
                target=CPUWorker.run,
                args=(
                    CPUWorker(
                        cpu_idx=cpu_idx,
                        exchanger=exchanger,
                        gpu_pipe_names=gpu_pipe_names,
                        lazy_collection_path=fp.name,
                        device=cpu_worker_devices[cpu_idx],
                    ),
                ),
            )
        )

    logging.info(f"Main PID {os.getpid()}")

    logging.info(
        f"Starting {num_cpu_workers} cpu workers and {num_gpu_workers} gpu workers on "
        f"{gpu_worker_devices}, with accelerated pipes: {gpu_pipe_names}",
    )

    for worker in (*cpu_workers, *gpu_workers):
        worker.start()

    logging.info("Workers are ready")

    for i in range(len((*cpu_workers, *gpu_workers))):
        outputs, count, cpu_idx, output_task_id = exchanger.outputs_queue.get()
        if isinstance(outputs, BaseException):
            raise outputs

    os.unlink(fp.name)

    num_max_enqueued = 1
    # Number of input/output batch per process
    outputs_iterator = exchanger.iter_results()

    cpu_worker_indices = list(range(num_cpu_workers))
    inputs_iterator = lc.reader.read_main()
    active_chunks = [{} for i in cpu_worker_indices]
    non_finalized = {i for i in cpu_worker_indices}
    max_workload = lc.chunk_size * num_max_enqueued

    bar = nullcontext()
    if show_progress:
        from tqdm import tqdm

        bar = tqdm(smoothing=0.1, mininterval=5.0)

    def get_and_process_output():
        outputs, count, cpu_idx, output_task_id = next(outputs_iterator)
        if output_task_id == "finalize":
            non_finalized.discard(cpu_idx)
        if isinstance(outputs, BaseException):
            raise outputs
        if show_progress:
            bar.update(count)
        if count > 0:
            yield outputs
        if output_task_id is not None:
            active_chunks[cpu_idx].pop(output_task_id, None)

    def process():
        try:
            with bar, lc.eval():
                for input_task_id, items in enumerate(
                    batchify(
                        iterable=inputs_iterator,
                        batch_size=lc.chunk_size,
                        drop_last=False,
                        formula=lambda x: sum(item[1] for item in x),
                    )
                ):
                    batch = [item[0] for item in items]
                    batch_size = sum(item[1] for item in items)

                    while all(sum(wl.values()) >= max_workload for wl in active_chunks):
                        yield from get_and_process_output()

                    # Shuffle to ensure the first process does not receive all the
                    # documents in case of workload equality
                    shuffle(cpu_worker_indices)
                    cpu_idx = min(
                        cpu_worker_indices,
                        key=lambda i: sum(active_chunks[i].values()),
                    )
                    exchanger.put_cpu((input_task_id, batch), stage=0, idx=cpu_idx)
                    active_chunks[cpu_idx][input_task_id] = batch_size

                # Inform the CPU workers that there are no more tasks to process
                for i, worker in enumerate(cpu_workers):
                    exchanger.cpu_inputs_queues[i][0].put(None)

                while any(active_chunks):
                    yield from get_and_process_output()

                while len(non_finalized):
                    yield from get_and_process_output()

        finally:
            revert_pickler()

            for k, v in old_environ.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

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
                    logging.error(f"Killing <GPUWorker idx={i}>")
                    worker.kill()
            for i, worker in enumerate(cpu_workers):  # pragma: no cover
                if worker.is_alive():
                    logging.error(f"Killing <CPUWorker idx={i}>")
                    worker.kill()

            for queue_group in (
                *exchanger.cpu_inputs_queues,
                *exchanger.gpu_inputs_queues,
                [exchanger.outputs_queue],
            ):
                for queue in queue_group:
                    if hasattr(queue, "cancel_join_thread"):
                        queue.cancel_join_thread()

    gen = process()
    return lc.writer.write_main(gen) if lc.writer is not None else flatten_once(gen)
