import pytest
from spacy.tokens import Span

from edsnlp.data.converters import (
    FILENAME,
    get_dict2doc_converter,
    get_doc2dict_converter,
)


def test_read_omop_dict(blank_nlp):
    json = {
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
    }
    doc = get_dict2doc_converter(
        "omop",
        dict(
            nlp=blank_nlp,
            span_attributes="negation",
            bool_attributes="negation",
        ),
    )[0](json)
    assert doc.text == "This is a test."
    assert doc._.note_id == 1234
    assert len(doc.ents) == 2
    assert doc.ents[0].text == "This"
    assert doc.ents[0]._.negation is True
    assert doc.ents[1]._.negation is False


def test_read_standoff_dict(blank_nlp):
    json = {
        "doc_id": 1234,
        "text": "This is a test.",
        "entities": [
            {
                "entity_id": 0,
                "fragments": [
                    {
                        "begin": 0,
                        "end": 4,
                    }
                ],
                "attributes": {
                    "negation": True,
                },
                "label": "test",
            },
            {
                "entity_id": 1,
                "fragments": [
                    {
                        "begin": 5,
                        "end": 7,
                    }
                ],
                "attributes": {},
                "label": "test",
            },
        ],
    }
    doc = get_dict2doc_converter(
        "standoff",
        dict(
            nlp=blank_nlp,
            span_attributes={"negation": "negation"},
            bool_attributes="negation",
        ),
    )[0](json)
    assert doc.text == "This is a test."
    assert doc._.note_id == 1234
    assert len(doc.ents) == 2
    assert doc.ents[0].text == "This"
    assert doc.ents[0]._.negation is True
    assert doc.ents[1]._.negation is False


def test_write_omop_dict(blank_nlp):
    doc = blank_nlp("This is a test.")
    doc._.note_id = 1234
    doc.ents = [Span(doc, 0, 1, label="test"), Span(doc, 1, 2, label="test")]
    doc.ents[0]._.negation = True
    doc.ents[1]._.negation = False
    json = {
        FILENAME: 1234,
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
                "negation": False,
            },
        ],
    }
    assert (
        get_doc2dict_converter(
            "omop",
            dict(
                span_getter={"ents": True},
                span_attributes=["negation"],
            ),
        )[0](doc)
        == json
    )


def test_write_standoff_dict(blank_nlp):
    doc = blank_nlp("This is a test.")
    doc._.note_id = 1234
    doc.ents = [Span(doc, 0, 1, label="test"), Span(doc, 1, 2, label="test")]
    if not Span.has_extension("negation"):
        Span.set_extension("negation", default=None)
    doc.ents[0]._.negation = True
    doc.ents[1]._.negation = False
    json = {
        FILENAME: 1234,
        "doc_id": 1234,
        "text": "This is a test.",
        "entities": [
            {
                "entity_id": 0,
                "fragments": [
                    {
                        "begin": 0,
                        "end": 4,
                    }
                ],
                "attributes": {
                    "negation": True,
                },
                "label": "test",
            },
            {
                "entity_id": 1,
                "fragments": [
                    {
                        "begin": 5,
                        "end": 7,
                    }
                ],
                "attributes": {
                    "negation": False,
                },
                "label": "test",
            },
        ],
    }
    assert (
        get_doc2dict_converter(
            "standoff",
            dict(
                span_getter={"ents": True},
                span_attributes={"negation": "negation"},
            ),
        )[0](doc)
        == json
    )


def test_unknown_converter():
    with pytest.raises(ValueError):
        get_dict2doc_converter("test", {})

    with pytest.raises(ValueError):
        get_doc2dict_converter("test", {})


def test_callable_converter():
    raw = lambda x: x  # noqa: E731
    assert get_dict2doc_converter(raw, {}) == (raw, {})
    assert get_doc2dict_converter(raw, {}) == (raw, {})
