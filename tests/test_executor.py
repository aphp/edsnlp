import asyncio

import pytest
import spacy.tokens
import xxhash

import edsnlp
import edsnlp.pipes as eds
from edsnlp.core.stream import Batchable, BatchifyOp, MapBatchesOp, Stream, UnbatchifyOp
from edsnlp.utils.batching import batchify, batchify_by_fragment, batchify_by_length_sum

try:
    import foldedtensor as ft
    import torch

    from edsnlp.pipes.trainable.embeddings.typing import WordEmbeddingComponent
except ImportError:  # pragma: no cover
    ft = torch = None
    WordEmbeddingComponent = object


class DumbEmbedding(WordEmbeddingComponent):
    output_size = 4
    span_getter = None

    def __init__(self, nlp=None, name="dumb_embedding"):
        super().__init__(nlp, name)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def preprocess(self, doc, *, contexts=None, **kwargs):
        contexts = contexts if contexts is not None else [doc[:]]
        return {
            "hashes": [
                [
                    [
                        xxhash.xxh32_intdigest(token.text.lower(), seed=idx)
                        for idx in range(self.output_size)
                    ]
                    for token in context
                ]
                for context in contexts
            ],
        }

    def collate(self, batch):
        contexts = [context for sample in batch["hashes"] for context in sample]

        indices = ft.as_folded_tensor(
            [list(range(len(context))) for context in contexts],
            data_dims=("word",),
            full_names=("context", "word"),
            dtype=torch.long,
        )
        vectors = torch.as_tensor(
            [
                [
                    ((word_hash >> (8 * dim)) & 0xFF) / 255
                    for dim, word_hash in enumerate(word_hashes)
                ]
                for context in contexts
                for word_hashes in context
            ],
            dtype=torch.float,
        )
        return {"embeddings": indices.with_data(vectors).refold("context", "word")}

    def forward(self, batch):
        return {"embeddings": batch["embeddings"]}


def prepare_gpu_error_batch(batch, device):
    return list(batch)


def raise_on_bad_gpu_batch(batch):
    if "bad" in batch:
        raise ValueError("bad gpu batch")
    return {"items": [item.upper() for item in batch]}


def postprocess_gpu_error_batch(docs, result, inputs=None):
    return result["items"]


def uppercase_batch(batch):
    return [item.upper() for item in batch]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_executor_rejects_readerless_stream():
    with pytest.raises(ValueError, match="from_queue"):
        Stream().executor()


@pytest.mark.anyio
async def test_executor_simple_backend():
    async with edsnlp.data.from_queue().map(lambda x: x.upper()).executor() as executor:
        fut1 = executor.submit("foo")
        fut2 = executor.submit("bar")
        assert await asyncio.gather(fut1, fut2) == ["FOO", "BAR"]


@pytest.mark.anyio
async def test_executor_item_error():
    def raise_on_bad(item):
        if item == "bad":
            raise ValueError("bad item")
        return item.upper()

    async with edsnlp.data.from_queue().map(raise_on_bad).executor() as executor:
        good = executor.submit("foo")
        bad = executor.submit("bad")
        later = executor.submit("bar")

        assert await good == "FOO"
        with pytest.raises(ValueError, match="bad item"):
            await bad
        assert await later == "BAR"
        assert await executor.submit("baz") == "BAZ"


@pytest.mark.anyio
async def test_executor_batch_error():
    def raise_on_bad_batch(batch):
        if "bad" in batch:
            raise ValueError("bad batch")
        return [item.upper() for item in batch]

    stream = Stream(
        reader=edsnlp.data.from_queue().reader,
        ops=[
            BatchifyOp(4, batchify),
            MapBatchesOp(
                Batchable(raise_on_bad_batch),
                {},
                elementwise=True,
            ),
            UnbatchifyOp(),
        ],
    )
    async with stream.executor() as executor:
        futures = [
            executor.submit(item) for item in ["a", "bad", "c", "d", "e", "f", "g", "h"]
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)

    assert [type(result) for result in results[:4]] == [ValueError] * 4
    assert [str(result) for result in results[:4]] == ["bad batch"] * 4
    assert results[4:] == ["E", "F", "G", "H"]


@pytest.mark.anyio
async def test_executor_multiprocessing_item_error():
    def raise_on_bad(item):
        if item == "bad":
            raise ValueError("bad item")
        return item.upper()

    stream = (
        edsnlp.data.from_queue()
        .map(raise_on_bad)
        .set_processing(
            backend="multiprocessing",
            num_cpu_workers=2,
            deterministic=False,
        )
    )
    async with stream.executor() as executor:
        results = await asyncio.gather(
            executor.submit("foo"),
            executor.submit("bad"),
            executor.submit("bar"),
            return_exceptions=True,
        )
        assert results[0] == "FOO"
        assert isinstance(results[1], ValueError)
        assert str(results[1]) == "bad item"
        assert results[2] == "BAR"
        assert await executor.submit("baz") == "BAZ"


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore:Using fork start method with GPU workers")
@pytest.mark.parametrize(
    "processing",
    [
        {"backend": "simple"},
        {
            "backend": "multiprocessing",
            "num_cpu_workers": 1,
            "num_gpu_workers": 1,
            "gpu_worker_devices": ["cpu"],
            "deterministic": False,
            "process_start_method": "spawn",
        },
    ],
    ids=["simple", "multiprocessing"],
)
async def test_executor_gpu_forward_error(processing):
    pytest.importorskip("torch")

    stream = (
        edsnlp.data.from_queue()
        .map_gpu(
            prepare_gpu_error_batch,
            raise_on_bad_gpu_batch,
            postprocess=postprocess_gpu_error_batch,
            batch_size=4,
        )
        .set_processing(**processing)
    )
    async with stream.executor() as executor:
        results = await asyncio.gather(
            *[executor.submit(item) for item in ["a", "bad", "c", "d"]],
            return_exceptions=True,
        )
        assert [type(result) for result in results] == [ValueError] * 4
        assert {
            str(result) for result in results if isinstance(result, ValueError)
        } == {"bad gpu batch"}
        results = await asyncio.gather(
            *[executor.submit(item) for item in ["e", "f", "g", "h"]],
            return_exceptions=True,
        )
        assert results == ["E", "F", "G", "H"]


@pytest.mark.anyio
async def test_executor_batch_wait_timeout():
    stream = Stream(
        reader=edsnlp.data.from_queue().reader,
        ops=[
            BatchifyOp(4, batchify),
            MapBatchesOp(
                Batchable(lambda batch: [item.upper() for item in batch]),
                {},
                elementwise=True,
            ),
            UnbatchifyOp(),
        ],
    )
    executor = stream.executor()
    try:
        futures = [executor.submit(str(i)) for i in range(18)]
        done, pending = await asyncio.wait(futures, timeout=1.0)
        assert len(done) == 16
        assert len(pending) == 2
    finally:
        await executor.aclose()
    assert await asyncio.gather(*futures) == [str(i).upper() for i in range(18)]

    async with stream.executor(batch_wait_timeout=0.01) as executor:
        futures = [executor.submit(str(i)) for i in range(18)]
        assert await asyncio.gather(*futures) == [str(i).upper() for i in range(18)]


@pytest.mark.anyio
async def test_executor_multiprocessing_length_batch_wait_timeout_flushes_tail():
    stream = Stream(
        reader=edsnlp.data.from_queue().reader,
        ops=[
            BatchifyOp(10, batchify_by_length_sum),
            MapBatchesOp(
                Batchable(uppercase_batch),
                {},
                elementwise=True,
            ),
            UnbatchifyOp(),
        ],
    ).set_processing(
        backend="multiprocessing",
        num_cpu_workers=2,
        deterministic=False,
    )

    async with stream.executor(batch_wait_timeout=0.01) as executor:
        futures = [executor.submit("aa") for _ in range(18)]
        results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)

    assert results == ["AA"] * 18


@pytest.mark.anyio
async def test_executor_map_pipeline():
    nlp = edsnlp.blank("eds")
    async with (
        edsnlp.data.from_queue()
        .map_pipeline(nlp)
        .map(lambda doc: doc.text)
        .executor() as executor
    ):
        assert await executor.submit("hello") == "hello"


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore:Using fork start method with GPU workers")
async def test_executor_map_torch_component_deterministic_fragments():
    pytest.importorskip("torch")

    if not spacy.tokens.Span.has_extension("cui"):
        spacy.tokens.Span.set_extension("cui", default=None)

    words = ["folfox", "doliprane", "doliprone", "patiente", "docteur"]

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.span_linker(
            rescale=20.0,
            threshold=0.0,
            metric="cosine",
            reference_mode="synonym",
            probability_mode="softmax",
            span_getter=["ents"],
            context_getter=["ents"],
            embedding=eds.span_pooler(
                hidden_size=None,
                embedding=DumbEmbedding(),
            ),
        ),
        name="linker",
    )

    def convert(entry):
        doc = nlp.make_doc(entry["STR"].lower()[:100])
        span = spacy.tokens.Span(doc, 0, len(doc), label=entry["GRP"])
        span._.cui = entry["CUI"]
        doc.ents = [span]
        return doc

    synonyms = edsnlp.data.from_iterable(
        [{"STR": word, "CUI": word, "GRP": "ENTITY"} for word in words],
        converter=convert,
    )
    nlp.post_init(synonyms)

    def dict_to_doc(entry):
        doc = nlp.make_doc(entry["text"])
        doc.ents = [spacy.tokens.Span(doc, 0, len(doc), label="ENTITY")]
        return doc

    def doc_to_dict(doc):
        ent = doc.ents[0]
        return {"text": doc.text, "cui": ent._.cui}

    samples = [{"text": w} for _ in range(8) for w in words]

    stream = (
        edsnlp.data.from_queue()
        .map(dict_to_doc)
        .map_pipeline(nlp, batch_size="10 words")
        .map(doc_to_dict)
        .set_processing(
            backend="multiprocessing",
            num_cpu_workers=2,
            num_gpu_workers=1,
            gpu_worker_devices=["cpu"],
            deterministic=True,
            process_start_method="spawn",
        )
    )
    # without batch_wait_timeout this would fail because the batcher would expect full
    # batches of 10 but the second batch is partial and the client waits for it to come
    # back before sending the 3rd batch, so it would keep waiting indefinitely
    async with stream.executor(batch_wait_timeout=0.01) as executor:
        futures = []
        for start, stop in [(0, 18), (18, 20), (20, len(samples))]:
            futures.extend(executor.submit(sample) for sample in samples[start:stop])
            await asyncio.sleep(0.02)

        results = await asyncio.wait_for(asyncio.gather(*futures), timeout=10.0)

    assert results == [
        {
            "text": sample["text"],
            "cui": sample["text"],
        }
        for sample in samples
    ]


def test_nondeterministic_processing_rejects_sentinel_preservation():
    stream = Stream(
        ops=[BatchifyOp(2, batchify, sentinel_mode="split")]
    ).set_processing(
        backend="multiprocessing",
        num_cpu_workers=2,
        deterministic=False,
    )

    with pytest.raises(ValueError, match="sentinel preservation"):
        stream._make_stages(split_torch_pipes=False)


def test_nondeterministic_processing_rejects_sentinel_requirements():
    stream = Stream(ops=[BatchifyOp(None, batchify_by_fragment)]).set_processing(
        backend="multiprocessing",
        num_cpu_workers=2,
        deterministic=False,
    )

    with pytest.raises(ValueError, match="require sentinel values"):
        stream._make_stages(split_torch_pipes=False)
