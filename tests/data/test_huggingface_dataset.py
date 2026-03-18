from typing import Iterator, TypeVar

import datasets
import pytest

import edsnlp
from edsnlp.data.huggingface_dataset import (
    from_huggingface_dataset,
    to_huggingface_dataset,
)

T = TypeVar("T")


def test_from_huggingface_dataset_conll2003_requires_split_when_omitted():
    with pytest.raises(ValueError, match=r"contains multiple splits"):
        # Use empty converter string to avoid triggering converter validation.
        from_huggingface_dataset(
            "lhoestq/conll2003",
            converter="",
            load_kwargs={"streaming": True},
        )


def test_from_huggingface_dataset_conll2003_from_dataset_wrong_split_name():
    conll2003 = datasets.load_dataset(
        "lhoestq/conll2003",
    )
    with pytest.raises(ValueError, match=r"Cannot select split"):
        # Use empty converter string to avoid triggering converter validation.
        from_huggingface_dataset(
            conll2003,
            split="dev",
            converter="",
            load_kwargs={"streaming": True},
        )


def test_from_huggingface_dataset_conll2003_from_dataset_name_wrong_split_name():
    with pytest.raises(ValueError, match=r"Could not load dataset"):
        # Use empty converter string to avoid triggering converter validation.
        from_huggingface_dataset(
            "lhoestq/conll2003",
            split="dev",
            converter="",
            load_kwargs={"streaming": True},
        )


def test_from_huggingface_dataset_conll2003_yields_records_without_converter():
    ds_dict = datasets.load_dataset("lhoestq/conll2003", streaming=True)
    stream = from_huggingface_dataset(ds_dict, split="train", converter="")
    it: Iterator[dict] = iter(stream)

    item = next(it)
    assert isinstance(item, dict)
    assert "tokens" in item
    assert "ner_tags" in item
    assert isinstance(item["tokens"], list)


def test_from_huggingface_dataset_conll2003_hf_text_validation_raises():
    with pytest.raises(ValueError, match=r"Cannot find these columns.*text_column"):
        from_huggingface_dataset(
            "lhoestq/conll2003",
            split="train",
            converter="hf_text",
            load_kwargs={"streaming": True},
        )


def test_from_huggingface_dataset_conll2003_hf_ner_converter_produces_docs():
    from spacy.tokens import Doc

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
    for x in stream:
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


def test_from_huggingface_dataset_bilou_schema():
    from spacy.tokens import Doc

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
    from spacy.tokens import Doc

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
        assert [ent.label_ for ent in doc1.ents] == [ent.label_ for ent in doc2.ents]


def test_from_huggingface_dataset_imdb_requires_split_when_omitted():
    with pytest.raises(ValueError, match=r"contains multiple splits"):
        # Use empty converter string to avoid triggering converter validation.
        from_huggingface_dataset(
            "imdb",
            converter="",
            load_kwargs={"streaming": True},
        )


def test_from_huggingface_dataset_imdb_yields_records_without_converter():
    ds_dict = datasets.load_dataset("imdb", streaming=True)
    stream = from_huggingface_dataset(ds_dict, split="train", converter="")
    it: Iterator[dict] = iter(stream)

    item = next(it)
    assert isinstance(item, dict)
    assert "text" in item
    assert "label" in item
    assert isinstance(item["text"], str)


def test_from_huggingface_dataset_imdb_hf_ner_converter_validation_raises():
    with pytest.raises(ValueError, match=r"Cannot find these columns.*words_column"):
        from_huggingface_dataset(
            "imdb",
            split="train",
            converter="hf_ner",
            load_kwargs={"streaming": True},
        )


def test_from_huggingface_dataset_imdb_hf_text_converter_produces_docs():
    from spacy.tokens import Doc

    nlp = edsnlp.blank("eds")
    stream = from_huggingface_dataset(
        "imdb",
        split="train",
        converter="hf_text",
        nlp=nlp,
        load_kwargs={"streaming": True},
    )

    docs = []
    for x in stream:
        docs.append(x)
        if len(docs) >= 2:
            break

    assert len(docs) == 2
    for doc in docs:
        assert isinstance(doc, Doc)
        assert doc.text.strip()
        assert len(doc) > 0


def test_from_huggingface_dataset_imdb_hf_text_converter_shuffle_reproducibility():
    from spacy.tokens import Doc

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


def test_from_huggingface_dataset_wikipedia_hf_text_converter_without_id_column():
    from spacy.tokens import Doc

    nlp = edsnlp.blank("eds")
    stream = from_huggingface_dataset(
        "wikimedia/wikipedia",
        name="20231101.ady",
        split="train",
        converter="hf_text",
        text_column="text",
        nlp=nlp,
        load_kwargs={"streaming": True},
    )

    docs = []
    for x in stream:
        docs.append(x)
        if len(docs) >= 2:
            break

    assert len(docs) == 2
    for doc in docs:
        assert isinstance(doc, Doc)
        assert doc.text.strip()
        assert len(doc) > 0


def test_huggingface_dataset_imdb_roundtrip():
    from datasets import IterableDataset

    nlp = edsnlp.blank("eds")
    stream = from_huggingface_dataset(
        "imdb",
        split="train",
        converter="hf_text",
        nlp=nlp,
        load_kwargs={"streaming": True},
    )

    docs = []
    for x in stream:
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


def test_huggingface_dataset_conll2003_roundtrip():
    from datasets import load_dataset

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


def test_from_huggingface_dataset_looping():
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
