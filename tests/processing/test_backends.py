import random
import time
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import pytest
from confit import VisibleDeprecationWarning, validate_arguments
from spacy.tokens import Doc

import edsnlp.data
import edsnlp.processing
from edsnlp.core.stream import Stream
from edsnlp.data.converters import get_current_tokenizer
from edsnlp.processing.multiprocessing import (
    MultiprocessingStreamExecutor,
    batch_from_bytes,
    batch_to_bytes,
    get_dispatch_schedule,
)

pytestmark = pytest.mark.processing

try:
    import torch.nn
except ImportError:
    torch = None


docs = [
    {
        "note_id": 1234,
        "note_text": "This is a test.",
        "entities": [
            {
                "note_nlp_id": 0,
                "start_char": 0,
                "end_char": 4,
                "lexical_variant": "This",
                "note_nlp_source_value": "test",
                "negation": True,
            },
            {
                "note_nlp_id": 1,
                "start_char": 5,
                "end_char": 7,
                "lexical_variant": "is",
                "note_nlp_source_value": "test",
            },
        ],
    },
    {
        "note_id": 0,
        "note_text": "This is an empty document.",
        "entities": None,
    },
]


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_tensor_byte_transport_preserves_layout():
    class TensorWithoutNames(torch.Tensor):
        @property
        def names(self):
            raise AttributeError("names")

    tensors = [
        torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16)[:, ::2],
        torch.empty((0, 3)),
        torch.tensor(3),
        torch.tensor([1]).expand(4, 1),
        torch.arange(4).as_subclass(TensorWithoutNames),
    ]
    supports_named_tensors = hasattr(torch.Tensor, "refine_names")
    if supports_named_tensors:
        tensors.append(torch.arange(6).reshape(2, 3).refine_names("row", "column"))

    restored = batch_from_bytes(batch_to_bytes(tensors))

    assert all(
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.stride() == expected.stride()
        and torch.equal(actual, expected)
        for actual, expected in zip(restored, tensors)
    )
    if supports_named_tensors:
        assert restored[-1].names == tensors[-1].names


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_tensor_byte_transport_preserves_folded_tensor():
    ft = pytest.importorskip("foldedtensor")
    tensor = ft.as_folded_tensor([[1.0, 2.0], [3.0]]).to(torch.bfloat16)

    restored = batch_from_bytes(batch_to_bytes(tensor))

    assert restored.dtype == tensor.dtype
    assert restored.lengths == tensor.lengths
    assert restored.data_dims == tensor.data_dims
    assert restored.full_names == tensor.full_names
    assert torch.equal(restored, tensor)


def make_gpu_stream():
    return Stream().map_gpu(
        prepare_batch=lambda docs, device: docs,
        forward=lambda batch: batch,
        postprocess=lambda docs, result, inputs=None: docs,
    )


def test_processing_assignment_config():
    stream = Stream().set_processing(preserve_output_order=False)

    assert stream.worker_assignment == "auto"
    assert stream.deterministic is False
    assert stream.preserve_output_order is False

    for deterministic, assignment in [(True, "static"), (False, "dynamic")]:
        with pytest.warns(
            VisibleDeprecationWarning,
            match="deterministic is deprecated",
        ):
            stream = Stream().set_processing(deterministic=deterministic)
        assert stream.worker_assignment == assignment
        assert stream.preserve_output_order is deterministic

    with pytest.raises(ValueError, match="both deterministic and worker_assignment"):
        Stream().set_processing(deterministic=False, worker_assignment="dynamic")
    with pytest.raises(ValueError, match="cannot preserve input order"):
        Stream().set_processing(
            worker_assignment="dynamic",
            preserve_output_order=True,
        )


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_default_cpu_workers_are_stage_agnostic(monkeypatch):
    monkeypatch.setenv("EDSNLP_MAX_CPU_WORKERS", "16")

    single_stage = make_gpu_stream().set_processing(num_gpu_workers=1)
    multi_stage = single_stage.map_gpu(
        prepare_batch=lambda docs, device: docs,
        forward=lambda batch: batch,
        postprocess=lambda docs, result, inputs=None: docs,
    )

    assert MultiprocessingStreamExecutor.adjust_num_workers(single_stage)[0] == 12
    assert MultiprocessingStreamExecutor.adjust_num_workers(multi_stage)[0] == 12


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_multiprocessing_device_cpu_disables_gpu_workers(monkeypatch):
    monkeypatch.setenv("EDSNLP_MAX_CPU_WORKERS", "4")

    stream = make_gpu_stream().set_processing(
        backend="multiprocessing",
        num_cpu_workers=1,
        device="cpu",
    )

    assert MultiprocessingStreamExecutor.adjust_num_workers(stream)[:4] == (
        1,
        0,
        ["cpu"],
        [],
    )


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_multiprocessing_rejects_preserve_device():
    stream = make_gpu_stream().set_processing(
        backend="multiprocessing",
        device="preserve",
    )

    with pytest.raises(ValueError, match="preserve"):
        MultiprocessingStreamExecutor.adjust_num_workers(stream)


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_multiprocessing_explicit_cuda_device(monkeypatch):
    monkeypatch.setenv("EDSNLP_MAX_CPU_WORKERS", "4")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    stream = make_gpu_stream().set_processing(
        backend="multiprocessing",
        device="cuda:0",
    )

    assert MultiprocessingStreamExecutor.adjust_num_workers(stream)[1:4] == (
        1,
        ["cpu", "cpu"],
        ["cuda:0"],
    )

    stream = make_gpu_stream().set_processing(
        backend="multiprocessing",
        device="cuda:0",
        num_gpu_workers=2,
    )

    with pytest.raises(ValueError, match="explicit CUDA device"):
        MultiprocessingStreamExecutor.adjust_num_workers(stream)


@pytest.mark.parametrize(
    "reader_format,reader_converter,backend,writer_format,writer_converter,worker_io",
    [
        ("pandas", "omop", "simple", "pandas", "omop", False),
        ("pandas", "omop", "multiprocessing", "pandas", "omop", False),
        pytest.param(
            "pandas",
            "omop",
            "spark",
            "pandas",
            "omop",
            False,
            marks=pytest.mark.processing,
            id="pandas-spark-pandas",
        ),
        ("parquet", "omop", "simple", "parquet", "omop", False),
        ("parquet", "omop", "multiprocessing", "parquet", "omop", False),
        pytest.param(
            "parquet",
            "omop",
            "spark",
            "parquet",
            "omop",
            False,
            marks=pytest.mark.processing,
            id="parquet-spark-parquet",
        ),
        ("parquet", "omop", "multiprocessing", "parquet", "omop", True),
        pytest.param(
            "parquet",
            "omop",
            "spark",
            "parquet",
            "omop",
            True,
            marks=pytest.mark.processing,
            id="parquet-spark-parquet-worker-io",
        ),
        ("parquet", "omop", "multiprocessing", "iterable", None, False),
    ],
)
def test_end_to_end(
    reader_format,
    reader_converter,
    backend,
    writer_format,
    writer_converter,
    worker_io,
    nlp_eds,
    tmp_path,
):
    nlp = nlp_eds
    rsrc = Path(__file__).parent.parent.resolve() / "resources"
    if reader_format == "pandas":
        pandas_dataframe = pd.DataFrame(docs)
        data = edsnlp.data.from_pandas(
            pandas_dataframe,
            converter=reader_converter,
        )
    elif reader_format == "parquet":
        data = edsnlp.data.read_parquet(
            rsrc / "docs.parquet",
            converter=reader_converter,
            read_in_worker=worker_io,
        )
    else:
        raise ValueError(reader_format)

    data = data.map_batches(lambda x: sorted(x, key=len), batch_size=2)
    data = data.map_pipeline(nlp)

    data = data.set_processing(
        backend=backend,
        show_progress=True,
        batch_by="words",
        batch_size=2,
    )

    if writer_format == "pandas":
        data.to_pandas(converter=writer_converter)
    elif writer_format == "parquet":
        data.write_parquet(
            tmp_path,
            converter=writer_converter,
            write_in_worker=worker_io,
        )
    elif writer_format == "iterable":
        list(data)
    else:
        raise ValueError(writer_format)


@pytest.mark.ml
def test_multiprocessing_backend(frozen_ml_nlp):
    texts = ["Ceci est un exemple", "Ceci est un autre exemple"]
    docs = list(
        frozen_ml_nlp.pipe(
            texts * 20,
            batch_size=2,
        ).set_processing(
            backend="multiprocessing",
            num_cpu_workers=-1,
            sort_chunks=True,
            chunk_size=2,
            batch_by="words",
            show_progress=True,
        )
    )
    assert len(docs) == 40


def error_pipe(doc: Doc):
    if doc._.note_id == "text-3":
        raise ValueError("error")
    return doc


@pytest.mark.ml
@pytest.mark.parametrize(
    "backend,worker_assignment,preserve_output_order",
    [
        ("simple", None, None),
        ("multiprocessing", "static", True),
        ("multiprocessing", "dynamic", False),
        ("spark", None, None),
    ],
)
def test_multiprocessing_gpu_stub_backend(
    frozen_ml_nlp,
    backend,
    worker_assignment,
    preserve_output_order,
):
    text1 = "Ceci est un exemple"
    text2 = "Ceci est un autre exemple"
    stream = frozen_ml_nlp.pipe(
        chain.from_iterable(
            [
                text1,
                text2,
            ]
            for i in range(2)
        ),
    )
    if backend == "simple":
        pass
    elif backend == "multiprocessing":
        stream = stream.set_processing(
            batch_size=2,
            num_gpu_workers=1,
            num_cpu_workers=1,
            gpu_worker_devices=["cpu"],
            worker_assignment=worker_assignment,
            preserve_output_order=preserve_output_order,
        )
    elif backend == "spark":
        stream = stream.set_processing(backend="spark")
    list(stream)


@pytest.mark.ml
def test_multiprocessing_gpu_stub_multi_cpu_static_backend(frozen_ml_nlp):
    text1 = "Exemple"
    text2 = "Ceci est un autre exemple"
    text3 = "Ceci est un très long exemple ! Regardez tous ces mots !"
    texts = [text1, text2, text3] * 100
    random.Random(42).shuffle(texts)
    stream = frozen_ml_nlp.pipe(iter(texts))
    stream = stream.set_processing(
        batch_size="15 words",
        num_gpu_workers=1,
        num_cpu_workers=2,
        worker_assignment="static",
        preserve_output_order=True,
        # show_progress=True,
        # just to test in gpu-less environments
        gpu_worker_devices=["cpu"],
    )
    list(stream)


@pytest.mark.parametrize("wait", [True, False])
@pytest.mark.ml
def test_multiprocessing_gpu_stub_wait(frozen_ml_nlp, wait):
    text1 = "Ceci est un exemple"
    text2 = "Ceci est un autre exemple"
    it = iter(
        frozen_ml_nlp.pipe(
            chain.from_iterable(
                [
                    text1,
                    text2,
                ]
                for i in range(2)
            ),
        ).set_processing(
            batch_size=2,
            num_gpu_workers=1,
            num_cpu_workers=1,
            gpu_worker_devices=["cpu"],
        )
    )
    if wait:
        time.sleep(5)
    list(it)


def simple_converter(obj):
    tok = get_current_tokenizer()
    doc = tok(obj["content"])
    doc._.note_id = obj["id"]
    return doc


def test_iterable_error():
    class Gen:
        def __iter__(self):
            for i in range(5):
                if i == 3:
                    raise ValueError("error")
                yield {"content": f"text-{i}", "id": f"text-{i}"}

    with pytest.raises(ValueError):
        list(
            edsnlp.data.from_iterable(Gen(), converter=simple_converter).set_processing(
                num_cpu_workers=2
            )
        )


@pytest.mark.ml
def test_multiprocessing_rb_error(ml_nlp):
    text1 = "Ceci est un exemple"
    text2 = "Ceci est un autre exemple"
    ml_nlp.add_pipe(error_pipe, name="error", after="sentences")
    with pytest.raises(ValueError):
        docs = edsnlp.data.from_iterable(
            chain.from_iterable(
                [
                    {"content": text1, "id": f"text-{i}"},
                    {"content": text2, "id": f"other-text-{i}"},
                ]
                for i in range(5)
            ),
            converter=simple_converter,
        ).map(lambda x: time.sleep(0.2) or x)
        docs = ml_nlp.pipe(
            docs,
            n_process=2,
            batch_size=2,
        )
        list(docs)


if torch is not None:
    from edsnlp.core.torch_component import BatchInput, BatchOutput, TorchComponent

    class DeepLearningError(TorchComponent):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def preprocess(self, doc):
            return {"num_words": len(doc), "doc_id": doc._.note_id}

        def collate(self, batch):
            return {
                "num_words": torch.tensor(batch["num_words"]),
                "doc_id": batch["doc_id"],
            }

        def forward(self, batch):
            if "text-1" in batch["doc_id"]:
                raise RuntimeError("Deep learning error")
            return {}


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.ml
def test_multiprocessing_ml_error(ml_nlp):
    text1 = "Ceci est un exemple"
    text2 = "Ceci est un autre exemple"
    ml_nlp.add_pipe(
        DeepLearningError(pipeline=ml_nlp),
        name="error",
        after="sentences",
    )

    with pytest.raises(RuntimeError) as e:
        docs = edsnlp.data.from_iterable(
            chain.from_iterable(
                [
                    {"content": text1, "id": f"text-{i}"},
                    {"content": text2, "id": f"other-text-{i}"},
                ]
                for i in range(5)
            ),
            converter=simple_converter,
        )
        docs = ml_nlp.pipe(docs)
        docs = docs.set_processing(
            batch_size=2,
            num_gpu_workers=1,
            num_cpu_workers=1,
            gpu_worker_devices=["cpu"],
        )
        list(docs)
    assert "Deep learning error" in str(e.value)


@pytest.mark.parametrize(
    "backend",
    [
        "simple",
        "multiprocessing",
        pytest.param("spark", marks=pytest.mark.processing, id="spark"),
    ],
)
def test_generator(backend):
    items = ["abc", "def", "ghij"]
    items = edsnlp.data.from_iterable(items)

    def gen(x):
        for char in x:
            yield char

    items = items.map(gen).set_processing(backend=backend, num_cpu_workers=2)
    # output from workers will be read in a round-robin fashion
    # ie zip(
    #   ("a",      "b",      "c",      "g", "h", "i", "j")  # worker 1
    #        ("d",      "e",      "f")  # worker 2
    # )
    assert set(items) == {"a", "d", "b", "e", "c", "f", "g", "h", "i", "j"}


@pytest.mark.parametrize(
    "worker_assignment,preserve_output_order,ordered",
    [("static", True, True), ("dynamic", False, False), ("static", False, False)],
)
def test_multiprocessing_sleep(worker_assignment, preserve_output_order, ordered):
    def process(x):
        if x % 2 == 0:
            time.sleep(0.1)
        return x

    items = list(range(100))
    items = edsnlp.data.from_iterable(items)
    items = items.map(process)
    items = items.set_processing(
        backend="multiprocessing",
        worker_assignment=worker_assignment,
        preserve_output_order=preserve_output_order,
        num_cpu_workers=2,
    )
    items = list(items)
    if ordered:
        assert items == list(range(100))
    else:
        assert items != list(range(100))


@pytest.mark.parametrize("num_cpu_workers", [0, 1, 2])
def test_static_ordered_skip(num_cpu_workers):
    def process_batch(x):
        return [i for i in x if i < 10 or i % 2 == 0]

    items = list(range(100))
    items = edsnlp.data.from_iterable(items)
    items = items.map_batches(process_batch)
    items = items.set_processing(
        worker_assignment="static",
        preserve_output_order=True,
        num_cpu_workers=num_cpu_workers,
    )
    items = list(items)
    assert items == [*range(0, 10), *range(10, 100, 2)]


if torch is not None:

    @validate_arguments
    class CacheInnerComponent(TorchComponent):
        def __init__(self, nlp=None, *args, **kwargs):
            super().__init__()
            self.called_forward = False

        def preprocess(self, doc):
            return {"text": doc.text}

        def collate(self, batch: Dict[str, Any]) -> BatchInput:
            return {"sizes": torch.as_tensor([len(x) for x in batch["text"]])}

        def forward(self, batch):
            assert not self.called_forward
            self.called_forward = True
            return {"sizes": batch["sizes"] * 2}

    @validate_arguments
    class CacheOuterComponent(TorchComponent):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def preprocess(self, doc):
            return {"inner": self.inner.preprocess(doc)}

        def collate(self, batch: Dict[str, Any]) -> BatchInput:
            return {"inner": self.inner.collate(batch["inner"])}

        def forward(self, batch: BatchInput) -> BatchOutput:
            return {"inner": self.inner(batch["inner"])["sizes"].clone()}

        def postprocess(
            self,
            docs: Sequence[Doc],
            results: BatchOutput,
            inputs: List[Dict[str, Any]],
        ) -> Sequence[Doc]:
            return docs


@pytest.mark.parametrize(
    "backend",
    ["simple", "multiprocessing"],
)
@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.ml
def test_backend_cache(backend):
    from edsnlp.core.torch_component import _caches

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(CacheInnerComponent(), name="inner")
    nlp.add_pipe(CacheOuterComponent(nlp.pipes.inner), name="outer")
    text1 = "Word"
    text2 = "A phrase"
    text3 = "This is a sentence"
    text4 = "This is a longer document with many words."
    stream = edsnlp.data.from_iterable([text1, text2, text3, text4])
    stream = stream.map_pipeline(nlp)
    if backend == "simple":
        assert list(_caches) == []
        list(stream.set_processing(batch_size=4))
        assert list(_caches) == []
    elif backend == "multiprocessing":
        list(
            stream.set_processing(
                batch_size=2,
                num_gpu_workers=2,
                num_cpu_workers=1,
                gpu_worker_devices=["cpu", "cpu"],
            )
        )
    elif backend == "spark":
        list(stream.set_processing(backend="spark", batch_size=4))


def test_task_dispatch_schedule():
    fn = get_dispatch_schedule

    assert fn(0, range(4), range(2)) == [0, 0]
    assert fn(1, range(4), range(2)) == [1, 1]
    assert fn(2, range(4), range(2)) == [0, 0]
    assert fn(3, range(4), range(2)) == [1, 1]

    assert fn(0, range(3), range(2)) == [0, 0]
    assert fn(1, range(3), range(2)) == [1, 1]
    assert fn(2, range(3), range(2)) == [0, 1]

    assert fn(0, range(2), range(3)) == [0, 0, 2]
    assert fn(1, range(2), range(3)) == [1, 1, 2]
    assert fn(0, range(2), range(3)) == [0, 0, 2]
    assert fn(1, range(2), range(3)) == [1, 1, 2]

    assert fn(0, range(16), range(10)) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert fn(1, range(16), range(10)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert fn(2, range(16), range(10)) == [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    assert fn(3, range(16), range(10)) == [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    assert fn(4, range(16), range(10)) == [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    assert fn(5, range(16), range(10)) == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    assert fn(6, range(16), range(10)) == [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    assert fn(7, range(16), range(10)) == [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    assert fn(8, range(16), range(10)) == [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    assert fn(9, range(16), range(10)) == [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    assert fn(10, range(16), range(10)) == [0, 0, 0, 0, 0, 0, 6, 6, 6, 6]
    assert fn(11, range(16), range(10)) == [1, 1, 1, 1, 1, 1, 7, 7, 7, 7]
    assert fn(12, range(16), range(10)) == [2, 2, 2, 2, 2, 2, 8, 8, 8, 8]
    assert fn(13, range(16), range(10)) == [3, 3, 3, 3, 3, 3, 9, 9, 9, 9]
    assert fn(14, range(16), range(10)) == [4, 4, 4, 4, 4, 4, 6, 7, 6, 7]
    assert fn(15, range(16), range(10)) == [5, 5, 5, 5, 5, 5, 8, 9, 8, 9]


def test_multiprocessing_on_simple_iterable_in_main():
    exec(
        """
import edsnlp.data

counter = 0

def complex_func(n):
    global counter
    counter += 1
    return n * n

stream = edsnlp.data.from_iterable(range(20))
stream = stream.map(complex_func)
stream = stream.set_processing(num_cpu_workers=2)
res = list(stream)
""",
        {"__MODULE__": "__main__"},
    )


def test_multiprocessing_on_full_example_in_main():
    exec(
        """
from spacy.tokens import Doc

import edsnlp
import edsnlp.pipes as eds
from edsnlp.data.converters import get_current_tokenizer

if not Doc.has_extension("note_text"):
    Doc.set_extension("note_text", default=None)
if not Doc.has_extension("date"):
    Doc.set_extension("date", default=None)
if not Doc.has_extension("person_id"):
    Doc.set_extension("person_id", default=None)


def convert_row_to_doc(row):
    if row["note_text"] is None:
        row["note_text"] = ""
    text = row["note_text"]
    doc = get_current_tokenizer()(text)
    doc._.note_id = row["note_id"]
    return doc


def convert_doc_to_row(doc_):
    note_id = doc_._.note_id
    person_id = doc_._.person_id
    note_text = doc_.text
    result = []
    for date in doc_.spans["dates"]:
        result.append(
            {
                "note_id": note_id,
                "person_id": person_id,
                # "note_text" : note_text,
                # "note_doc" : doc_,
                "date": date._.date.datetime,
            }
        )
    return result


nlp = edsnlp.blank("eds")
# nlp = eds_biomedic_aphp.load()
# nlp.add_pipe(eds.sections())
nlp.add_pipe(eds.dates())
nlp.add_pipe(eds.sentences())
data = edsnlp.data.from_iterable(
    [{"note_text": "Test", "note_id": "test"}],
    converter=convert_row_to_doc,
)
data = data.map_pipeline(nlp)
data_pd = data.set_processing(show_progress=True, num_cpu_workers=5).to_pandas(
    converter=convert_doc_to_row
)
""",
        {"__MODULE__": "__main__"},
    )
