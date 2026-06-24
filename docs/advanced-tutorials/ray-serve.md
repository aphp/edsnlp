# Serving with Ray Serve

Ray Serve can host an EDS-NLP stream and keep the stream executor in charge of
batching and multiprocessing. This is useful when the same deployment should be
usable from HTTP and from Ray RPC.

Install the development dependency when working from the repository:

```console
uv add 'ray[serve]' 'protobuf<7' --dev
```

## Create a deployment

Compose the stream so that it receives and returns Python dictionaries. Use
EDS-NLP data converters at the serving boundary when the pipeline itself works
on spaCy documents.

```{ .python .no-check }
import edsnlp
import edsnlp.pipes as eds


nlp = edsnlp.blank("eds")
nlp.add_pipe(eds.normalizer())
nlp.add_pipe(eds.sentences())
nlp.add_pipe(
    eds.matcher(
        regex={"drug": ["doliprane", "paracetamol"]},
        attr="LOWER",
    )
)

stream = (
    edsnlp.data.from_queue(converter="hf_text", text_column="text")
    .map_pipeline(nlp)
    .to_iterable(converter="omop")
)

handle = stream.deploy_ray_serve(
    name="edsnlp",
    route_prefix="/process",
    batch_wait_timeout=0.05,
    deployment_options={
        "num_replicas": "auto",
        "ray_actor_options": {"num_cpus": 4},
        "max_ongoing_requests": 64,
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 8,
            "target_ongoing_requests": 32,
        },
    },
)
```

The deployment exposes one method for RPC:

```{ .python .no-check }
result = await handle.process.remote({"text": "Le patient prend du doliprane"})
```

The same deployment can receive one JSON item through HTTP:

```console
curl -X POST http://localhost:8000/process \
  -H 'content-type: application/json' \
  -d '{"text": "Le patient prend du doliprane"}'
```

## Processing resources

When the stream has no processing backend configured, Ray serving can infer a
multiprocessing backend from the Ray actor resources. For example, an actor with
`num_cpus=4` gets three CPU workers by default, leaving one CPU for the Ray
replica process, while an actor with `num_cpus=1` keeps the simple backend.
When a torch stream reserves `num_gpus=1` and `num_cpus=1`, it keeps the simple
backend and runs torch components on CUDA. When it reserves `num_gpus=1` and
`num_cpus=4`, it gets three CPU workers and one GPU worker. Without a Ray GPU
reservation, torch streams run on CPU by default.

You can still call `stream.set_processing(...)` before `deploy_ray_serve(...)` when
you need explicit worker counts or devices.

Pass `run=False` when you need the Ray deployment object and want to call
`serve.run(...)` yourself.

EDS-NLP executors use `batch_wait_timeout` for batch timeout flushing, meaning
that from the arrival of a given item, we won't wait and accumulate inputs in a
batch for more than this `batch_wait_timeout` amount, before sending the batch.
The  stream batching configuration controls the (max) size of batches processed
by the pipeline.

## Parameters

::: edsnlp.core.stream.Stream.deploy_ray_serve
    options:
        heading_level: 3
        show_source: false
