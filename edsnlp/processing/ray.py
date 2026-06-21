from __future__ import annotations

import asyncio
import warnings
from typing import Any, Optional

from pydantic import NonNegativeFloat

from edsnlp.core.stream import Stream


def infer_stream_processing(stream: Stream, deployment_options: dict) -> Stream:
    ray_actor_options = deployment_options.get("ray_actor_options") or {}

    resources = {}
    try:
        import ray

        resources = ray.get_runtime_context().get_assigned_resources()
    except Exception:  # pragma: no cover
        pass

    num_cpus = resources.get("CPU", ray_actor_options.get("num_cpus"))
    num_gpus = resources.get("GPU", ray_actor_options.get("num_gpus"))
    if stream.backend not in (None, "simple", "multiprocessing"):
        return stream

    kwargs = {}
    torch_components = list(stream.torch_components())
    has_torch_components = bool(torch_components)
    device = stream.device
    num_gpus_value = None if num_gpus is None else float(num_gpus)
    has_gpu_resources = num_gpus_value is not None and num_gpus_value > 0
    effective_num_cpus = (
        num_cpus if num_cpus is not None else 1 if has_gpu_resources else None
    )

    if has_gpu_resources and not has_torch_components:
        warnings.warn(
            "Ray actor reserves GPU resources, but this stream has no "
            "EDS-NLP torch/GPU components. EDS-NLP will not use the GPU; "
            "the GPU remains reserved by Ray and visible to the replica process.",
            stacklevel=2,
        )
    if has_gpu_resources and has_torch_components and device == "cpu":
        warnings.warn(
            "Ray actor reserves GPU resources, but stream processing is forced "
            "to device='cpu'. The GPU remains reserved by Ray and visible to the "
            "replica process.",
            stacklevel=2,
        )
    if has_gpu_resources and has_torch_components and device != "cpu":
        if not num_gpus_value.is_integer():
            warnings.warn(
                "Ray actor reserves a fractional GPU. EDS-NLP will use one "
                "visible GPU device; fractional sharing is controlled by Ray.",
                stacklevel=2,
            )
    if has_gpu_resources and has_torch_components and device == "auto":
        kwargs["device"] = "cuda"
    elif not has_gpu_resources and has_torch_components and device == "auto":
        kwargs["device"] = "cpu"

    inferred_cpu_workers = None
    if stream.backend in (None, "multiprocessing") and stream.num_cpu_workers is None:
        if effective_num_cpus is not None:
            inferred_cpu_workers = max(0, int(effective_num_cpus) - 1)

    inferred_gpu_workers = None
    if (
        stream.backend in (None, "multiprocessing")
        and stream.num_gpu_workers is None
        and has_gpu_resources
        and has_torch_components
        and device != "cpu"
        and (
            inferred_cpu_workers
            or stream.backend == "multiprocessing"
            or (stream.num_cpu_workers is not None and stream.num_cpu_workers > 0)
        )
    ):
        inferred_gpu_workers = max(1, int(num_gpus_value))

    if (
        has_gpu_resources
        and has_torch_components
        and num_gpus_value > 1
        and not inferred_gpu_workers
    ):
        warnings.warn(
            "Ray actor reserves multiple GPUs, but this stream will run in the "
            "simple backend and use one visible GPU. Set multiprocessing workers "
            "explicitly to use multiple GPUs.",
            stacklevel=2,
        )

    if inferred_cpu_workers is not None and (
        inferred_cpu_workers
        or inferred_gpu_workers
        or stream.backend == "multiprocessing"
    ):
        kwargs["num_cpu_workers"] = inferred_cpu_workers

    if inferred_gpu_workers:
        kwargs["num_gpu_workers"] = inferred_gpu_workers

    if stream.backend is None and (
        "num_cpu_workers" in kwargs or "num_gpu_workers" in kwargs
    ):
        kwargs["backend"] = "multiprocessing"

    return stream.set_processing(**kwargs) if kwargs else stream


def deploy_ray_serve(
    stream: Stream,
    *,
    name: str = "default",
    route_prefix: Optional[str] = "/",
    batch_wait_timeout: Optional[NonNegativeFloat] = None,
    deployment_options: Optional[dict] = None,
    infer_processing: bool = True,
    run: bool = True,
    start: bool = True,
    proxy_location=None,
    http_options: Optional[dict] = None,
    grpc_options: Optional[dict] = None,
    logging_config: Optional[dict] = None,
):
    from edsnlp.core.executor import StreamExecutor

    StreamExecutor.validate(stream)

    import ray
    from ray import serve
    from starlette.requests import Request

    deployment_options = dict(deployment_options or {})
    stream_ref = ray.put(stream) if run else None
    local_stream = None if run else stream

    async def run_executor_group(executor, items: list[Any]) -> list[Any]:
        if not items:
            return []
        futures = [executor.submit(item) for item in items]
        return await asyncio.gather(*futures)

    def flatten_requests(
        requests: list[Any],
    ) -> tuple[list[Any], list[tuple[int, int, bool]]]:
        items = []
        spans = []
        for request in requests:
            start = len(items)
            if isinstance(request, list):
                items.extend(request)
                spans.append((start, len(items), True))
            else:
                items.append(request)
                spans.append((start, len(items), False))
        return items, spans

    def restore_requests(
        values: list[Any],
        spans: list[tuple[int, int, bool]],
    ) -> list[Any]:
        return [
            values[start:end] if is_batch else values[start]
            for start, end, is_batch in spans
        ]

    @serve.deployment(**deployment_options)
    class StreamDeployment:
        def __init__(self):
            source_stream = (
                ray.get(stream_ref) if stream_ref is not None else local_stream
            )
            serving_stream = (
                infer_stream_processing(source_stream, deployment_options)
                if infer_processing
                else source_stream
            )
            if (
                serving_stream.backend == "multiprocessing"
                and serving_stream.process_start_method is None
            ):
                serving_stream = serving_stream.set_processing(
                    process_start_method="spawn",
                )
            self.executor = serving_stream.executor(
                batch_wait_timeout=batch_wait_timeout,
            )

        async def process_requests(self, requests: list[Any]) -> list[Any]:
            items, spans = flatten_requests(requests)
            results = await run_executor_group(self.executor, items)
            return restore_requests(results, spans)

        async def __call__(self, request: Request) -> Any:
            return await self.process(await request.json())

        async def process(self, item: Any) -> Any:
            return (await self.process_requests([item]))[0]

    if not run:
        return StreamDeployment

    if start:
        serve.start(
            proxy_location=proxy_location,
            http_options=http_options,
            grpc_options=grpc_options,
            logging_config=logging_config,
        )

    return serve.run(
        StreamDeployment.bind(),
        name=name,
        route_prefix=route_prefix,
        logging_config=logging_config,
    )
