import os
from typing import Iterator, TypeVar

import pytest

from edsnlp.data.huggingface_dataset import (
    from_huggingface_dataset,
    to_huggingface_dataset,
)

T = TypeVar("T")


def _skip_if_offline() -> None:
    if os.environ.get("HF_DATASETS_OFFLINE") in {"1", "true", "True"}:
        pytest.skip("HF_DATASETS_OFFLINE is enabled")
    if os.environ.get("TRANSFORMERS_OFFLINE") in {"1", "true", "True"}:
        pytest.skip("TRANSFORMERS_OFFLINE is enabled")


def _maybe_skip_hf_hub_failure(exc: Exception) -> None:
    msg = (str(exc) or "").lower()
    # Best-effort: if the hub is unreachable in CI, skip rather than fail.
    network_markers = [
        "connection",
        "timed out",
        "timeout",
        "temporary failure",
        "name or service not known",
        "getaddrinfo",
        "ssl",
        "proxy",
        "connection reset",
        "connection aborted",
        "max retries",
        "offline",
        "not connected",
        "503",
        "502",
        "504",
    ]
    if any(m in msg for m in network_markers):
        pytest.skip(f"HuggingFace Hub not reachable: {exc!r}")


def test_from_huggingface_dataset_conll2003_requires_split_when_omitted():
    pytest.importorskip("datasets")
    _skip_if_offline()

    try:
        with pytest.raises(ValueError, match=r"contains multiple splits"):
            # Use empty converter string to avoid triggering converter validation.
            from_huggingface_dataset(
                "lhoestq/conll2003",
                converter="",
                load_kwargs={"streaming": True},
            )
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_from_huggingface_dataset_conll2003_from_dataset_wrong_split_name():
    datasets = pytest.importorskip("datasets")
    _skip_if_offline()

    conll2003 = datasets.load_dataset(
        "lhoestq/conll2003",
    )
    try:
        with pytest.raises(ValueError, match=r"Cannot select split"):
            # Use empty converter string to avoid triggering converter validation.
            from_huggingface_dataset(
                conll2003,
                split="dev",
                converter="",
                load_kwargs={"streaming": True},
            )
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_from_huggingface_dataset_conll2003_from_dataset_name_wrong_split_name():
    pytest.importorskip("datasets")
    _skip_if_offline()

    try:
        with pytest.raises(ValueError, match=r"Could not load dataset"):
            # Use empty converter string to avoid triggering converter validation.
            from_huggingface_dataset(
                "lhoestq/conll2003",
                split="dev",
                converter="",
                load_kwargs={"streaming": True},
            )
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_from_huggingface_dataset_conll2003_yields_records_without_converter():
    datasets = pytest.importorskip("datasets")
    _skip_if_offline()

    try:
        ds_dict = datasets.load_dataset("lhoestq/conll2003", streaming=True)
        stream = from_huggingface_dataset(ds_dict, split="train", converter="")
        it: Iterator[dict] = stream.execute()

        item = next(it)
        assert isinstance(item, dict)
        assert "tokens" in item
        assert "ner_tags" in item
        assert isinstance(item["tokens"], list)
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_from_huggingface_dataset_conll2003_hf_text_validation_raises():
    pytest.importorskip("datasets")
    _skip_if_offline()

    try:
        with pytest.raises(ValueError, match=r"Cannot find these columns.*text_column"):
            from_huggingface_dataset(
                "lhoestq/conll2003",
                split="train",
                converter="hf_text",
                load_kwargs={"streaming": True},
            )
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_from_huggingface_dataset_conll2003_hf_ner_converter_produces_docs():
    pytest.importorskip("datasets")
    edsnlp = pytest.importorskip("edsnlp")
    from spacy.tokens import Doc

    _skip_if_offline()

    tag_order = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    ]

    try:
        nlp = edsnlp.blank("eds")
        stream = from_huggingface_dataset(
            "lhoestq/conll2003",
            split="train",
            converter="hf_ner",
            tag_order=tag_order,
            nlp=nlp,
            load_kwargs={"streaming": True},
        )

        docs = []
        for x in stream.execute():
            docs.append(x)
            if len(docs) >= 2:
                break

        assert len(docs) == 2
        for doc in docs:
            assert isinstance(doc, Doc)
            assert doc.text.strip()
            assert len(doc) > 0

            if hasattr(doc, "ents"):
                for ent in doc.ents:
                    assert ent.label_ in {"PER", "ORG", "LOC", "MISC"}
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_from_huggingface_dataset_bilou_schema():
    pytest.importorskip("datasets")
    edsnlp = pytest.importorskip("edsnlp")
    datasets = pytest.importorskip("datasets")
    from spacy.tokens import Doc

    _skip_if_offline()

    doc = {
        "tokens": ["John", "Doe", "lives", "in", "Paris", "since", "2020", "."],
        "ner_tags": ["B_PER", "E_PER", "O", "O", "LOC", "O", "B_DATE", "O"],
    }
    hf_dataset = datasets.Dataset.from_dict({k: [v] for k, v in doc.items()})
    dataset = edsnlp.data.from_huggingface_dataset(
        hf_dataset,
        converter="hf_ner",
    )

    # assert docs only has 2 entities
    docs = list(dataset.map_pipeline(edsnlp.blank("eds")))
    assert len(docs) == 1
    assert isinstance(docs[0], Doc)
    assert len(docs[0].ents) == 3


def test_from_huggingface_dataset_conll2003_hf_ner_converter_shuffle_reproducibility():
    pytest.importorskip("datasets")
    edsnlp = pytest.importorskip("edsnlp")
    from spacy.tokens import Doc

    _skip_if_offline()

    tag_order = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    ]
    tag_map = dict(enumerate(tag_order))

    try:
        nlp = edsnlp.blank("eds")
        stream1 = from_huggingface_dataset(
            "lhoestq/conll2003",
            split="train",
            converter="hf_ner",
            tag_map=tag_map,
            nlp=nlp,
            load_kwargs={"streaming": True},
            shuffle="dataset",
            seed=42,
        )

        stream2 = from_huggingface_dataset(
            "lhoestq/conll2003",
            split="train",
            converter="hf_ner",
            tag_map=tag_map,
            nlp=nlp,
            load_kwargs={"streaming": True},
            shuffle="dataset",
            seed=42,
        )

        docs1 = []
        docs2 = []
        for x in stream1.execute():
            docs1.append(x)
            if len(docs1) >= 5:
                break

        for x in stream2.execute():
            docs2.append(x)
            if len(docs2) >= 5:
                break

        assert len(docs1) == 5
        assert len(docs2) == 5

        for doc1, doc2 in zip(docs1, docs2):
            assert isinstance(doc1, Doc)
            assert isinstance(doc2, Doc)
            assert doc1.text == doc2.text
            assert [ent.label_ for ent in doc1.ents] == [
                ent.label_ for ent in doc2.ents
            ]
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_from_huggingface_dataset_imdb_requires_split_when_omitted():
    pytest.importorskip("datasets")
    _skip_if_offline()

    try:
        with pytest.raises(ValueError, match=r"contains multiple splits"):
            # Use empty converter string to avoid triggering converter validation.
            from_huggingface_dataset(
                "imdb",
                converter="",
                load_kwargs={"streaming": True},
            )
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_from_huggingface_dataset_imdb_yields_records_without_converter():
    datasets = pytest.importorskip("datasets")
    _skip_if_offline()

    try:
        ds_dict = datasets.load_dataset("imdb", streaming=True)
        stream = from_huggingface_dataset(ds_dict, split="train", converter="")
        it: Iterator[dict] = stream.execute()

        item = next(it)
        assert isinstance(item, dict)
        assert "text" in item
        assert "label" in item
        assert isinstance(item["text"], str)
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_from_huggingface_dataset_imdb_hf_ner_converter_validation_raises():
    pytest.importorskip("datasets")
    _skip_if_offline()

    try:
        with pytest.raises(
            ValueError, match=r"Cannot find these columns.*words_column"
        ):
            from_huggingface_dataset(
                "imdb",
                split="train",
                converter="hf_ner",
                load_kwargs={"streaming": True},
            )
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_from_huggingface_dataset_imdb_hf_text_converter_produces_docs():
    pytest.importorskip("datasets")
    edsnlp = pytest.importorskip("edsnlp")
    from spacy.tokens import Doc

    _skip_if_offline()

    try:
        nlp = edsnlp.blank("eds")
        stream = from_huggingface_dataset(
            "imdb",
            split="train",
            converter="hf_text",
            nlp=nlp,
            load_kwargs={"streaming": True},
        )

        docs = []
        for x in stream.execute():
            docs.append(x)
            if len(docs) >= 2:
                break

        assert len(docs) == 2
        for doc in docs:
            assert isinstance(doc, Doc)
            assert doc.text.strip()
            assert len(doc) > 0
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_from_huggingface_dataset_imdb_hf_text_converter_shuffle_reproducibility():
    pytest.importorskip("datasets")
    edsnlp = pytest.importorskip("edsnlp")
    from spacy.tokens import Doc

    _skip_if_offline()

    try:
        nlp = edsnlp.blank("eds")
        stream1 = from_huggingface_dataset(
            "imdb",
            split="train",
            converter="hf_text",
            nlp=nlp,
            load_kwargs={"streaming": True},
            shuffle="dataset",
            seed=42,
        )

        stream2 = from_huggingface_dataset(
            "imdb",
            split="train",
            converter="hf_text",
            nlp=nlp,
            load_kwargs={"streaming": True},
            shuffle="dataset",
            seed=42,
        )

        docs1 = []
        docs2 = []
        for x in stream1.execute():
            docs1.append(x)
            if len(docs1) >= 5:
                break

        for x in stream2.execute():
            docs2.append(x)
            if len(docs2) >= 5:
                break

        assert len(docs1) == 5
        assert len(docs2) == 5

        for doc1, doc2 in zip(docs1, docs2):
            assert isinstance(doc1, Doc)
            assert isinstance(doc2, Doc)
            assert doc1.text == doc2.text
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_huggingface_dataset_imdb_roundtrip():
    pytest.importorskip("datasets")
    edsnlp = pytest.importorskip("edsnlp")
    from datasets import IterableDataset

    _skip_if_offline()

    try:
        nlp = edsnlp.blank("eds")
        stream = from_huggingface_dataset(
            "imdb",
            split="train",
            converter="hf_text",
            nlp=nlp,
            load_kwargs={"streaming": True},
        )

        docs = []
        for x in stream.execute():
            docs.append(x)
            if len(docs) >= 5:
                break

        dataset = to_huggingface_dataset(
            docs,
            converter="hf_text",
            text_column="text",
        )

        assert isinstance(dataset, IterableDataset)
        list_dataset = list(dataset)
        assert len(list_dataset) == 5
        for item, doc in zip(dataset, docs):
            assert item["text"] == doc.text
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_huggingface_dataset_conll2003_roundtrip():
    pytest.importorskip("datasets")
    edsnlp = pytest.importorskip("edsnlp")
    from datasets import load_dataset

    _skip_if_offline()

    tag_order = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    ]

    try:
        nlp = edsnlp.blank("eds")
        dataset = load_dataset("lhoestq/conll2003", split="train")
        first_five = dataset.take(5)

        stream = from_huggingface_dataset(
            first_five,
            converter="hf_ner",
            tag_order=tag_order,
            nlp=nlp,
            load_kwargs={"streaming": True},
        )

        dataset = to_huggingface_dataset(
            list(stream),
            converter="hf_ner",
            words_column="tokens",
            ner_tags_column="ner_tags",
            execute=False,
        )
        list_dataset = list(dataset)
        assert len(list_dataset) == 5
        for item, original in zip(dataset, first_five):
            assert item["tokens"] == original["tokens"]
            assert item["ner_tags"] == [tag_order[i] for i in original["ner_tags"]]
    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise


def test_from_huggingface_dataset_looping():
    pytest.importorskip("datasets")
    edsnlp = pytest.importorskip("edsnlp")
    datasets = pytest.importorskip("datasets")

    _skip_if_offline()

    tag_order = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    ]

    try:
        nlp = edsnlp.blank("eds")
        dataset = datasets.load_dataset("lhoestq/conll2003", split="train[:10]")
        stream = from_huggingface_dataset(
            dataset,
            converter="hf_ner",
            tag_order=tag_order,
            nlp=nlp,
            load_kwargs={"streaming": True},
        )
        count = 0
        for _ in stream:
            count += 1
            if count == 10:
                break
        assert count == 10

        stream = from_huggingface_dataset(
            dataset,
            converter="hf_ner",
            tag_order=tag_order,
            nlp=nlp,
            load_kwargs={"streaming": True},
            loop=True,
        )
        count = 0
        for _ in stream:
            count += 1
            if count == 50:
                break
        assert count == 50

    except Exception as e:
        _maybe_skip_hf_hub_failure(e)
        raise
