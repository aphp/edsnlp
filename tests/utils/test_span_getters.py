import edsnlp
from edsnlp.utils.span_getters import make_span_context_getter


def test_span_context_getter_symmetric(lang):
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
