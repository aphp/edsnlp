import pytest
from confit import validate_arguments

import edsnlp
import edsnlp.pipes as eds
from edsnlp.utils.span_getters import (
    ContextWindow,
    get_spans,
    make_span_context_getter,
    validate_span_setter,
)


def test_span_getter_dedupliation(lang):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.diabetes")

    doc = nlp("le patient a un diabète")

    span_getter = {"diabetes": True, "ents": True}

    assert len(list(get_spans(doc, span_getter, deduplicate=False))) == 2
    assert len(list(get_spans(doc, span_getter, deduplicate=True))) == 1


def test_span_context_getter(lang):
    nlp = edsnlp.blank(lang)
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.matcher", config={"terms": {"sentence": "sentence"}})
    doc = nlp(
        "This is a sentence. "
        "This is another sentence. "
        "This is a third one. "
        "Last sentence."
    )

    span_getter = make_span_context_getter(
        context_words=2,
        context_sents=1,
    )
    assert [span_getter(s).text for s in doc.ents] == [
        "This is a sentence. This",
        "This is another sentence. This",
        ". Last sentence.",
    ]

    span_getter = make_span_context_getter(
        context_words=0,
        context_sents=1,
    )
    assert [span_getter(s).text for s in doc.ents] == [
        "This is a sentence.",
        "This is another sentence.",
        "Last sentence.",
    ]

    span_getter = make_span_context_getter(
        context_words=0,
        context_sents=2,
    )
    assert [span_getter(s).text for s in doc.ents] == [
        "This is a sentence. This is another sentence.",
        "This is a sentence. This is another sentence. This is a third one.",
        "This is a third one. Last sentence.",
    ]


def test_span_getter_on_span():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.sentences())
    nlp.add_pipe(
        eds.matcher(
            terms={"animal": ["snake", "dog"]},
            span_setter=["ents", "animals"],
        )
    )
    doc = nlp(
        "There was a snake. "
        "His friend was a dog. "
        "He liked baking cakes. "
        "But since he had no hands, he was a bad baker. "
    )
    sents = list(doc.sents)
    assert str(list(get_spans(sents[0], validate_span_setter("ents")))) == "[snake]"
    assert str(list(get_spans(sents[0], validate_span_setter("animals")))) == "[snake]"
    assert str(list(get_spans(doc[5:], validate_span_setter("animals")))) == "[dog]"
    assert str(list(get_spans(doc[5:], validate_span_setter("*")))) == "[dog]"


def test_span_context_getter_asymmetric(lang):
    nlp = edsnlp.blank(lang)
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.matcher", config={"terms": {"animal": "kangaroo"}})
    doc = nlp(
        "This is a sentence. "
        "This is another sentence with a kangaroo. "
        "This is a third one. "
        "Last sentence."
    )

    span_getter = make_span_context_getter(context_words=2, context_sents=0)
    assert [span_getter(s).text for s in doc.ents] == [
        "with a kangaroo. This",
    ]

    span_getter = make_span_context_getter(context_words=(2, 1), context_sents=0)
    assert [span_getter(s).text for s in doc.ents] == [
        "with a kangaroo.",
    ]

    span_getter = make_span_context_getter(context_words=(1, 2), context_sents=0)
    assert [span_getter(s).text for s in doc.ents] == [
        "a kangaroo. This",
    ]

    span_getter = make_span_context_getter(context_words=0, context_sents=(1, 2))
    assert [span_getter(s).text for s in doc.ents] == [
        "This is another sentence with a kangaroo. This is a third one.",
    ]

    span_getter = make_span_context_getter(context_words=0, context_sents=(2, 2))
    assert [span_getter(s).text for s in doc.ents] == [
        "This is a sentence. This is another sentence with a kangaroo. This is a third one."  # noqa: E501
    ]

    span_getter = make_span_context_getter(context_words=0, context_sents=(1, 1))
    assert [span_getter(s).text for s in doc.ents] == [
        "This is another sentence with a kangaroo."
    ]

    span_getter = make_span_context_getter(context_words=(1000, 0), context_sents=0)
    assert [span_getter(s).text for s in doc.ents] == [
        "This is a sentence. This is another sentence with a kangaroo"
    ]

    span_getter = make_span_context_getter(
        context_words=(1000, 0), context_sents=(1, 2)
    )
    assert [span_getter(s).text for s in doc.ents] == [
        "This is a sentence. This is another sentence with a kangaroo. This is a third one."  # noqa: E501
    ]


def test_context_getter_syntax():
    @validate_arguments
    def get_snippet(span, context: ContextWindow):
        return context(span)

    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.matcher", config={"terms": {"dog": "dog"}})
    doc = nlp(
        "There was a snake. "
        "His friend was a dog. "
        "He liked baking cakes. "
        "But since he had no hands, he was a bad baker. "
    )

    assert (
        get_snippet(doc.ents[0], "words[-5:5]").text
        == ". His friend was a dog. He liked baking cakes"
    )

    assert get_snippet(doc.ents[0], "words[-5:5] & sent").text == "His friend was a dog"

    assert (
        get_snippet(doc.ents[0], "words[-5:8] | sents[-1:1]").text
        == "There was a snake. His friend was a dog. He liked baking cakes. "
        "But since"
    )


def test_invalid_context_getter_syntax():
    @validate_arguments
    def apply_context(context: ContextWindow):
        pass

    apply_context("sents[-2:2]")

    with pytest.raises(ValueError):
        apply_context("stuff[-2:2]")
