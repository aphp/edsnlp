import sys
import types

import pytest

import edsnlp
from edsnlp.processing.ray import infer_stream_processing


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def ray_serve():
    ray = pytest.importorskip("ray")
    from ray import serve
    from ray.serve.config import ProxyLocation

    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    serve.start(proxy_location=ProxyLocation.Disabled)
    try:
        yield serve
    finally:
        serve.shutdown()
        ray.shutdown()


def test_ray_processing_inference_from_deployment_options():
    stream = edsnlp.data.from_queue()
    serving_stream = infer_stream_processing(
        stream,
        {"ray_actor_options": {"num_cpus": 2}},
    )

    assert serving_stream.backend == "multiprocessing"
    assert serving_stream.num_cpu_workers == 1

    stream = edsnlp.data.from_queue().set_processing(backend="spark")
    serving_stream = infer_stream_processing(
        stream,
        {"ray_actor_options": {"num_cpus": 2}},
    )

    assert serving_stream is stream


def test_ray_processing_inference_keeps_one_cpu_replica_simple():
    stream = edsnlp.data.from_queue()
    serving_stream = infer_stream_processing(
        stream,
        {"ray_actor_options": {"num_cpus": 1}},
    )

    assert serving_stream.backend is None
    assert serving_stream.num_cpu_workers is None


def test_ray_processing_inference_uses_cpu_without_reserved_gpu():
    stream = edsnlp.data.from_queue().map_gpu(
        prepare_batch=lambda batch, device: batch,
        forward=lambda batch: batch,
        batch_size=2,
    )
    serving_stream = infer_stream_processing(
        stream,
        {"ray_actor_options": {"num_cpus": 1}},
    )

    assert serving_stream.backend is None
    assert serving_stream.device == "cpu"


def test_ray_processing_inference_warns_on_gpu_without_torch_components():
    stream = edsnlp.data.from_queue()

    with pytest.warns(UserWarning, match="no EDS-NLP torch/GPU components"):
        serving_stream = infer_stream_processing(
            stream,
            {"ray_actor_options": {"num_cpus": 1, "num_gpus": 1}},
        )

    assert serving_stream.backend is None
    assert serving_stream.num_cpu_workers is None
    assert serving_stream.num_gpu_workers is None


def test_ray_processing_inference_warns_on_fractional_gpu():
    stream = edsnlp.data.from_queue().map_gpu(
        lambda batch, device: batch,
        lambda batch: batch,
        batch_size=2,
    )

    with pytest.warns(UserWarning, match="fractional GPU"):
        serving_stream = infer_stream_processing(
            stream,
            {"ray_actor_options": {"num_cpus": 1, "num_gpus": 0.5}},
        )

    assert serving_stream.backend is None
    assert serving_stream.num_cpu_workers is None
    assert serving_stream.num_gpu_workers is None
    assert serving_stream.device == "cuda"


def test_ray_processing_inference_uses_simple_cuda_with_one_cpu_and_one_gpu():
    stream = edsnlp.data.from_queue().map_gpu(
        prepare_batch=lambda batch, device: batch,
        forward=lambda batch: batch,
        batch_size=2,
    )
    serving_stream = infer_stream_processing(
        stream,
        {"ray_actor_options": {"num_cpus": 1, "num_gpus": 1}},
    )

    assert serving_stream.backend is None
    assert serving_stream.num_cpu_workers is None
    assert serving_stream.num_gpu_workers is None
    assert serving_stream.device == "cuda"


def test_ray_processing_inference_uses_simple_cuda_when_cpu_is_unset():
    stream = edsnlp.data.from_queue().map_gpu(
        prepare_batch=lambda batch, device: batch,
        forward=lambda batch: batch,
        batch_size=2,
    )
    serving_stream = infer_stream_processing(
        stream,
        {"ray_actor_options": {"num_gpus": 1}},
    )

    assert serving_stream.backend is None
    assert serving_stream.num_cpu_workers is None
    assert serving_stream.num_gpu_workers is None
    assert serving_stream.device == "cuda"


def test_ray_processing_inference_uses_gpu_workers_when_cpus_are_available():
    stream = edsnlp.data.from_queue().map_gpu(
        prepare_batch=lambda batch, device: batch,
        forward=lambda batch: batch,
        batch_size=2,
    )
    serving_stream = infer_stream_processing(
        stream,
        {"ray_actor_options": {"num_cpus": 4, "num_gpus": 1}},
    )

    assert serving_stream.backend == "multiprocessing"
    assert serving_stream.num_cpu_workers == 3
    assert serving_stream.num_gpu_workers == 1
    assert serving_stream.device == "cuda"


def test_ray_processing_inference_uses_gpu_workers_with_explicit_cpu_workers():
    stream = (
        edsnlp.data.from_queue()
        .map_gpu(
            prepare_batch=lambda batch, device: batch,
            forward=lambda batch: batch,
            batch_size=2,
        )
        .set_processing(num_cpu_workers=2)
    )
    serving_stream = infer_stream_processing(
        stream,
        {"ray_actor_options": {"num_gpus": 1}},
    )

    assert serving_stream.backend == "multiprocessing"
    assert serving_stream.num_cpu_workers == 2
    assert serving_stream.num_gpu_workers == 1
    assert serving_stream.device == "cuda"


def test_ray_serve_rejects_writer_stream(tmp_path):
    stream = edsnlp.data.from_queue().write_parquet(
        tmp_path / "out",
        batch_size=2,
        execute=False,
    )

    with pytest.raises(ValueError, match="writers"):
        stream.deploy_ray_serve(run=False)


class FakeServe:
    def __init__(self):
        self.started = None
        self.ran = None

    def deployment(self, **options):
        def decorate(cls):
            cls.deployment_options = options
            cls.bind = classmethod(lambda bound_cls: bound_cls())
            return cls

        return decorate

    def start(self, **kwargs):
        self.started = kwargs

    def run(self, deployment, **kwargs):
        self.ran = kwargs
        return deployment


def install_fake_ray_serve(monkeypatch):
    fake_serve = FakeServe()
    fake_ray = types.ModuleType("ray")
    fake_ray.serve = fake_serve
    fake_ray.put = lambda value: value
    fake_ray.get = lambda value: value
    fake_starlette = types.ModuleType("starlette")
    fake_requests = types.ModuleType("starlette.requests")
    fake_requests.Request = object
    fake_starlette.requests = fake_requests

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "starlette", fake_starlette)
    monkeypatch.setitem(sys.modules, "starlette.requests", fake_requests)
    return fake_serve


@pytest.mark.anyio
async def test_ray_serve_starts_and_processes_requests_with_fake_serve(monkeypatch):
    fake_serve = install_fake_ray_serve(monkeypatch)

    def uppercase_item(item):
        return {"text": item["text"].upper()}

    stream = edsnlp.data.from_queue().map(uppercase_item)
    deployment = stream.deploy_ray_serve(
        name="edsnlp_ray_fake",
        route_prefix=None,
        deployment_options={"ray_actor_options": {"num_cpus": 1}},
        batch_wait_timeout=0.01,
        infer_processing=False,
        start=True,
        proxy_location="disabled",
        http_options={"host": "127.0.0.1"},
        logging_config={"encoding": "TEXT"},
    )

    class Request:
        async def json(self):
            return {"text": "codeine"}

    try:
        assert fake_serve.started == {
            "proxy_location": "disabled",
            "http_options": {"host": "127.0.0.1"},
            "grpc_options": None,
            "logging_config": {"encoding": "TEXT"},
        }
        assert fake_serve.ran == {
            "name": "edsnlp_ray_fake",
            "route_prefix": None,
            "logging_config": {"encoding": "TEXT"},
        }
        assert await deployment.process_requests([]) == []
        assert await deployment.process({"text": "doliprane"}) == {"text": "DOLIPRANE"}
        assert await deployment.process(
            [{"text": "doliprane"}, {"text": "aspirine"}]
        ) == [{"text": "DOLIPRANE"}, {"text": "ASPIRINE"}]
        assert await deployment(Request()) == {"text": "CODEINE"}
    finally:
        deployment.executor.close()


@pytest.mark.anyio
async def test_ray_serve_returns_handle(ray_serve):
    def uppercase_item(item):
        return {"text": item["text"].upper()}

    stream = edsnlp.data.from_queue().map(uppercase_item)
    handle = stream.deploy_ray_serve(
        name="edsnlp_ray_test",
        route_prefix=None,
        deployment_options={"ray_actor_options": {"num_cpus": 1}},
        infer_processing=False,
        start=False,
    )

    assert await handle.process.remote({"text": "doliprane"}) == {"text": "DOLIPRANE"}


@pytest.mark.anyio
async def test_ray_serve_accepts_client_side_batches(ray_serve):
    def uppercase_item(item):
        return {"text": item["text"].upper()}

    stream = edsnlp.data.from_queue().map(uppercase_item)
    Deployment = stream.deploy_ray_serve(
        deployment_options={"ray_actor_options": {"num_cpus": 1}},
        infer_processing=False,
        run=False,
    )

    handle = ray_serve.run(
        Deployment.bind(),
        name="edsnlp_ray_client_batch_test",
        route_prefix=None,
    )

    assert await handle.process.remote(
        [{"text": "doliprane"}, {"text": "aspirine"}]
    ) == [{"text": "DOLIPRANE"}, {"text": "ASPIRINE"}]
