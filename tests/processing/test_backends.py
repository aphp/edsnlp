import time
from itertools import chain
from pathlib import Path

import pandas as pd
import pytest
from spacy.tokens import Doc

import edsnlp.data
import edsnlp.processing
from edsnlp.data.converters import get_current_tokenizer

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


@pytest.mark.parametrize(
    "reader_format,reader_converter,backend,writer_format,writer_converter,worker_io",
    [
        ("pandas", "omop", "simple", "pandas", "omop", False),
        ("pandas", "omop", "multiprocessing", "pandas", "omop", False),
        ("pandas", "omop", "spark", "pandas", "omop", False),
        ("parquet", "omop", "simple", "parquet", "omop", False),
        ("parquet", "omop", "multiprocessing", "parquet", "omop", False),
        ("parquet", "omop", "spark", "parquet", "omop", False),
        ("parquet", "omop", "multiprocessing", "parquet", "omop", True),
        ("parquet", "omop", "spark", "parquet", "omop", True),
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
            rsrc / "docs.pq",
            converter=reader_converter,
            read_in_worker=worker_io,
        )
    else:
        raise ValueError(reader_format)

    data = data.map_pipeline(nlp)

    data = data.set_processing(
        backend=backend,
        show_progress=True,
        chunk_size=2,
        batch_by="words",
        batch_size=2,
        sort_chunks=True,
    )

    if writer_format == "pandas":
        data.to_pandas(converter=writer_converter)
    elif writer_format == "parquet":
        if backend == "spark":
            with pytest.raises(ValueError):
                data.write_parquet(
                    tmp_path,
                    converter=writer_converter,
                    write_in_worker=worker_io,
                )
            data.write_parquet(
                tmp_path,
                converter=writer_converter,
                accumulate=False,
                write_in_worker=worker_io,
            )
        else:
            data.write_parquet(
                tmp_path,
                converter=writer_converter,
                write_in_worker=worker_io,
            )
    else:
        raise ValueError(writer_format)


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


def test_multiprocessing_gpu_stub(frozen_ml_nlp):
    text1 = "Ceci est un exemple"
    text2 = "Ceci est un autre exemple"
    list(
        frozen_ml_nlp.pipe(
            chain.from_iterable(
                [
                    text1,
                    text2,
                ]
                for i in range(5)
            ),
        ).set_processing(
            batch_size=2,
            num_gpu_workers=1,
            num_cpu_workers=1,
            gpu_worker_devices=["cpu"],
        )
    )


def simple_converter(obj):
    tok = get_current_tokenizer()
    doc = tok(obj["content"])
    doc._.note_id = obj["id"]
    return doc


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


try:
    import torch

    from edsnlp.core.torch_component import TorchComponent

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

except (ImportError, AttributeError):
    pass


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
