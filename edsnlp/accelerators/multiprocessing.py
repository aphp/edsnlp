import gc
import signal
from multiprocessing.connection import wait
from random import shuffle
from typing import Any, Iterable, List, Optional, Union

import torch
import torch.multiprocessing as mp

from edsnlp.core.registry import registry
from edsnlp.core.torch_component import TorchComponent
from edsnlp.utils.collections import batchify

from .base import Accelerator, FromDictFieldsToDoc, FromDoc, ToDoc

DEBUG = True

debug = (
    (lambda *args, flush=False, **kwargs: print(*args, **kwargs, flush=True))
    if DEBUG
    else lambda *args, **kwargs: None
)


class Exchanger:
    def __init__(
        self,
        num_stages,
        num_gpu_workers,
        num_cpu_workers,
        gpu_worker_devices,
    ):
        # queue for cpu input tasks
        self.gpu_worker_devices = gpu_worker_devices
        # We add prioritized queue at the end for STOP signals
        self.cpu_inputs_queues = [
            [mp.SimpleQueue()] + [mp.SimpleQueue() for _ in range(num_stages + 1)]
            # The input queue is not shared between processes, since calling `wait`
            # on a queue reader from multiple processes may lead to a deadlock
            for _ in range(num_cpu_workers)
        ]
        self.gpu_inputs_queues = [
            [mp.SimpleQueue() for _ in range(num_stages + 1)]
            for _ in range(num_gpu_workers)
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
                continue
            if item is None:
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


class CPUWorker(mp.Process):
    def __init__(
        self,
        cpu_idx: int,
        exchanger: Exchanger,
        gpu_pipe_names: List[str],
        model: Any,
        device: Union[str, torch.device],
    ):
        super(CPUWorker, self).__init__()

        self.cpu_idx = cpu_idx
        self.exchanger = exchanger
        self.gpu_pipe_names = gpu_pipe_names
        self.model = model
        self.device = device

    def _run(self):
        # Cannot pass torch tensor during init i think ? otherwise i get
        # ValueError: bad value(s) in fds_to_keep
        mp._prctl_pr_set_pdeathsig(signal.SIGINT)

        model = self.model.to(self.device)
        stages = [{"cpu_components": [], "gpu_component": None}]
        for name, component in model.pipeline:
            if name in self.gpu_pipe_names:
                stages[-1]["gpu_component"] = component
                stages.append({"cpu_components": [], "gpu_component": None})
            else:
                stages[-1]["cpu_components"].append(component)

        next_batch_id = 0
        active_batches = {}
        debug(
            f"CPU worker {self.cpu_idx} is ready",
            next(model.parameters()).device,
            flush=True,
        )

        had_error = False
        with torch.no_grad():
            for stage, task in self.exchanger.get_cpu_tasks(self.cpu_idx):
                if had_error:
                    continue  # pragma: no cover
                try:
                    if stage == 0:
                        gpu_idx = None
                        batch_id = next_batch_id
                        debug("preprocess start for", batch_id)
                        next_batch_id += 1
                        docs = task
                    else:
                        gpu_idx, batch_id, result = task
                        debug("postprocess start for", batch_id)
                        docs = active_batches.pop(batch_id)
                        gpu_pipe = stages[stage - 1]["gpu_component"]
                        docs = gpu_pipe.postprocess(docs, result)  # type: ignore

                    for component in stages[stage]["cpu_components"]:
                        if hasattr(component, "batch_process"):
                            docs = component.batch_process(docs)
                        else:
                            docs = [component(doc) for doc in docs]

                    gpu_pipe = stages[stage]["gpu_component"]
                    if gpu_pipe is not None:
                        preprocessed = gpu_pipe.make_batch(docs)  # type: ignore
                        active_batches[batch_id] = docs
                        if gpu_idx is None:
                            gpu_idx = batch_id % len(self.exchanger.gpu_worker_devices)
                        collated = gpu_pipe.collate(  # type: ignore
                            preprocessed,
                        )
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
                        debug("preprocess end for", batch_id)
                    else:
                        self.exchanger.put_results((docs, self.cpu_idx, gpu_idx))
                        debug("postprocess end for", batch_id)
                except BaseException as e:
                    had_error = True
                    import traceback

                    print(traceback.format_exc(), flush=True)
                    self.exchanger.put_results((e, self.cpu_idx, None))
            # We need to drain the queues of GPUWorker fed inputs (pre-moved to GPU)
            # to ensure no tensor allocated on producer processes (CPUWorker via
            # collate) are left in consumer processes
            debug("Start draining CPU worker", self.cpu_idx)
            [None for _ in self.exchanger.get_cpu_tasks(self.cpu_idx)]
        debug(f"CPU worker {self.cpu_idx} is about to stop")

    def run(self):
        self._run()
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


class GPUWorker(mp.Process):
    def __init__(
        self,
        gpu_idx,
        exchanger: Exchanger,
        gpu_pipe_names: List[str],
        model: Any,
        device: Union[str, torch.device],
    ):
        super().__init__()

        self.device = device
        self.gpu_idx = gpu_idx
        self.exchanger = exchanger

        self.gpu_pipe_names = gpu_pipe_names
        self.model = model
        self.device = device

    def _run(self):
        debug("GPU worker", self.gpu_idx, "started")
        mp._prctl_pr_set_pdeathsig(signal.SIGINT)
        had_error = False

        model = self.model.to(self.device)
        stage_components = [model.get_pipe(name) for name in self.gpu_pipe_names]
        del model
        with torch.no_grad():
            for stage, task in self.exchanger.get_gpu_tasks(self.gpu_idx):
                if had_error:
                    continue  # pragma: no cover
                try:
                    cpu_idx, batch_id, batch = task
                    debug("forward start for", batch_id)
                    component = stage_components[stage]
                    res = component.module_forward(batch)
                    del batch, task
                    # TODO set non_blocking=True here
                    res = {
                        key: val.to("cpu") if hasattr(val, "to") else val
                        for key, val in res.items()
                    }
                    self.exchanger.put_cpu(
                        item=(self.gpu_idx, batch_id, res),
                        stage=stage + 1,
                        idx=cpu_idx,
                    )
                    debug("forward end for", batch_id)
                except BaseException as e:
                    had_error = True
                    self.exchanger.put_results((e, None, self.gpu_idx))
                    import traceback

                    print(traceback.format_exc(), flush=True)
                task = batch = res = None  # noqa
            # We need to drain the queues of CPUWorker fed inputs (pre-moved to GPU)
            # to ensure no tensor allocated on producer processes (CPUWorker via
            # collate) are left in consumer processes
            debug("Start draining GPU worker", self.gpu_idx)
            [None for _ in self.exchanger.get_gpu_tasks(self.gpu_idx)]
        debug(f"GPU worker {self.gpu_idx} is about to stop")

    def run(self):
        self._run()
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


DEFAULT_MAX_CPU_WORKERS = 4


@registry.accelerator.register("multiprocessing")
class MultiprocessingAccelerator(Accelerator):
    """
    If you have multiple CPU cores, and optionally multiple GPUs, we provide a
    `multiprocessing` accelerator that allows to run the inference on multiple
    processes.

    This accelerator dispatches the batches between multiple workers
    (data-parallelism), and distribute the computation of a given batch on one or two
    workers (model-parallelism). This is done by creating two types of workers:

    - a `CPUWorker` which handles the non deep-learning components and the
      preprocessing, collating and postprocessing of deep-learning components
    - a `GPUWorker` which handles the forward call of the deep-learning components

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

    Examples
    --------

    ```{ .python .no-check }
    docs = list(
        nlp.pipe(
            [text1, text2, ...],
            accelerator={
                "@accelerator": "multiprocessing",
                "num_cpu_workers": 3,
                "num_gpu_workers": 2,
                "batch_size": 8,
            },
        )
    )
    ```

    Parameters
    ----------
    batch_size: int
        Number of documents to process at a time in a CPU/GPU worker
    num_cpu_workers: int
        Number of CPU workers. A CPU worker handles the non deep-learning components
        and the preprocessing, collating and postprocessing of deep-learning components.
    num_gpu_workers: Optional[int]
        Number of GPU workers. A GPU worker handles the forward call of the
        deep-learning components.
    gpu_pipe_names: Optional[List[str]]
        List of pipe names to accelerate on a GPUWorker, defaults to all pipes
        that inherit from TorchComponent
    """

    def __init__(
        self,
        batch_size: int,
        num_cpu_workers: Optional[int] = None,
        num_gpu_workers: Optional[int] = None,
        gpu_pipe_names: Optional[List[str]] = None,
        gpu_worker_devices: Optional[List[Union[torch.device, str]]] = None,
        cpu_worker_devices: Optional[List[Union[torch.device, str]]] = None,
    ):
        self.batch_size = batch_size
        self.num_gpu_workers: Optional[int] = num_gpu_workers
        self.num_cpu_workers = num_cpu_workers
        self.gpu_pipe_names = gpu_pipe_names
        self.gpu_worker_devices = gpu_worker_devices
        self.cpu_worker_devices = cpu_worker_devices

    def __call__(
        self,
        inputs: Iterable[Any],
        nlp: Any,
        to_doc: ToDoc = FromDictFieldsToDoc("content"),
        from_doc: FromDoc = lambda doc: doc,
    ):
        """
        Stream of documents to process. Each document can be a string or a tuple

        Parameters
        ----------
        inputs
        nlp

        Yields
        ------
        Any
            Processed outputs of the pipeline
        """
        if torch.multiprocessing.get_start_method() != "spawn":
            torch.multiprocessing.set_start_method("spawn", force=True)

        gpu_pipe_names = (
            [
                name
                for name, component in nlp.pipeline
                if isinstance(component, TorchComponent)
            ]
            if self.gpu_pipe_names is None
            else self.gpu_pipe_names
        )

        if not all(nlp.has_pipe(name) for name in gpu_pipe_names):
            raise ValueError(
                "GPU accelerated pipes {} could not be found in the model".format(
                    sorted(set(nlp.pipe_names) - set(gpu_pipe_names))
                )
            )

        num_devices = torch.cuda.device_count()
        print(f"Number of available devices: {num_devices}", flush=True)

        num_cpu_workers = self.num_cpu_workers
        num_gpu_workers = self.num_gpu_workers

        if num_gpu_workers is None:
            num_gpu_workers = num_devices if len(gpu_pipe_names) > 0 else 0

        if num_cpu_workers is None:
            num_cpu_workers = max(
                min(mp.cpu_count() - num_gpu_workers, DEFAULT_MAX_CPU_WORKERS), 0
            )

        if num_gpu_workers == 0:
            gpu_pipe_names = []

        gpu_worker_devices = (
            [
                torch.device(f"cuda:{gpu_idx * num_devices // num_gpu_workers}")
                for gpu_idx in range(num_gpu_workers)
            ]
            if self.gpu_worker_devices is None
            else self.gpu_worker_devices
        )
        cpu_worker_devices = (
            ["cpu"] * num_cpu_workers
            if self.cpu_worker_devices is None
            else self.cpu_worker_devices
        )
        assert len(cpu_worker_devices) == num_cpu_workers
        assert len(gpu_worker_devices) == num_gpu_workers
        if num_cpu_workers == 0:
            (
                num_cpu_workers,
                num_gpu_workers,
                cpu_worker_devices,
                gpu_worker_devices,
                gpu_pipe_names,
            ) = (num_gpu_workers, 0, gpu_worker_devices, [], [])

        debug(f"Number of CPU workers: {num_cpu_workers}")
        debug(f"Number of GPU workers: {num_gpu_workers}")

        exchanger = Exchanger(
            num_stages=len(gpu_pipe_names),
            num_cpu_workers=num_cpu_workers,
            num_gpu_workers=num_gpu_workers,
            gpu_worker_devices=gpu_worker_devices,
        )

        cpu_workers = []
        gpu_workers = []
        nlp = nlp.to("cpu")

        for gpu_idx in range(num_gpu_workers):
            gpu_workers.append(
                GPUWorker(
                    gpu_idx=gpu_idx,
                    exchanger=exchanger,
                    gpu_pipe_names=gpu_pipe_names,
                    model=nlp,
                    device=gpu_worker_devices[gpu_idx],
                )
            )

        for cpu_idx in range(num_cpu_workers):
            cpu_workers.append(
                CPUWorker(
                    cpu_idx=cpu_idx,
                    exchanger=exchanger,
                    gpu_pipe_names=gpu_pipe_names,
                    model=nlp,
                    device=cpu_worker_devices[cpu_idx],
                )
            )

        for worker in (*cpu_workers, *gpu_workers):
            worker.start()

        try:
            num_max_enqueued = num_cpu_workers * 2 + 10
            # Number of input/output batch per process
            total_inputs = [0] * num_cpu_workers
            total_outputs = [0] * num_cpu_workers
            outputs_iterator = exchanger.iter_results()

            cpu_worker_indices = list(range(num_cpu_workers))
            inputs_iterator = (to_doc(i, nlp) for i in inputs)
            for i, pdfs_batch in enumerate(batchify(inputs_iterator, self.batch_size)):
                if sum(total_inputs) - sum(total_outputs) >= num_max_enqueued:
                    outputs, cpu_idx, gpu_idx = next(outputs_iterator)
                    if isinstance(outputs, BaseException):
                        raise outputs  # pragma: no cover
                    yield from (from_doc(o) for o in outputs)
                    total_outputs[cpu_idx] += 1

                # Shuffle to ensure the first process does not receive all the documents
                # in case of total_inputs - total_outputs equality
                shuffle(cpu_worker_indices)
                cpu_idx = min(
                    cpu_worker_indices,
                    key=lambda i: total_inputs[i] - total_outputs[i],
                )
                exchanger.put_cpu(pdfs_batch, stage=0, idx=cpu_idx)
                total_inputs[cpu_idx] += 1

            while sum(total_outputs) < sum(total_inputs):
                outputs, cpu_idx, gpu_idx = next(outputs_iterator)
                if isinstance(outputs, BaseException):
                    raise outputs  # pragma: no cover
                yield from (from_doc(o) for o in outputs)
                total_outputs[cpu_idx] += 1
        finally:
            # Send gpu and cpu process the order to stop processing data
            # We use the prioritized queue to ensure the stop signal is processed
            # before the next batch of data
            for i, worker in enumerate(gpu_workers):
                exchanger.gpu_inputs_queues[i][-1].put(None)
                debug("Asked gpu worker", i, "to stop processing data")
            for i, worker in enumerate(cpu_workers):
                exchanger.cpu_inputs_queues[i][-1].put(None)
                debug("Asked cpu worker", i, "to stop processing data")

            # Enqueue a final non prioritized STOP signal to ensure there remains no
            # data in the queues (cf drain loop in CPUWorker / GPUWorker)
            for i, worker in enumerate(gpu_workers):
                exchanger.gpu_inputs_queues[i][0].put(None)
                debug("Asked gpu", i, "to end")
            for i, worker in enumerate(gpu_workers):
                worker.join(timeout=5)
                debug("Joined gpu worker", i)
            for i, worker in enumerate(cpu_workers):
                exchanger.cpu_inputs_queues[i][0].put(None)
                debug("Asked cpu", i, "to end")
            for i, worker in enumerate(cpu_workers):
                worker.join(timeout=1)
                debug("Joined cpu worker", i)

            # If a worker is still alive, kill it
            # This should not happen, but for a reason I cannot explain, it does in
            # some CPU workers sometimes when we catch an error, even though each run
            # method of the workers completes cleanly. Maybe this has something to do
            # with the cleanup of these processes ?
            for i, worker in enumerate(gpu_workers):  # pragma: no cover
                if worker.is_alive():
                    print("Killing gpu worker", i)
                    worker.kill()
            for i, worker in enumerate(cpu_workers):  # pragma: no cover
                if worker.is_alive():
                    print("Killing cpu worker", i)
                    worker.kill()
