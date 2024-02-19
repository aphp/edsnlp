import edsnlp
from edsnlp.utils.span_getters import make_span_context_getter


def test_span_sentence_getter(lang):
    nlp = edsnlp.blank("eds")
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
        span_getter=["ents"],
        context_words=2,
        overlap_policy="merge",
    )
    spans = span_getter(doc)
    assert [s.text for s in spans] == [
        "This is a sentence. This is another sentence. This",
        ". Last sentence.",
    ]

    span_getter = make_span_context_getter(
        span_getter=["ents"],
        context_words=2,
        overlap_policy="filter",
    )
    spans = span_getter(doc)
    assert [s.text for s in spans] == [
        "This is a sentence. This",
        ". Last sentence.",
    ]

    span_getter = make_span_context_getter(
        span_getter=["ents"],
        context_words=0,
        context_sents=1,
        overlap_policy="filter",
    )
    spans = span_getter(doc)
    assert [s.text for s in spans] == [
        "This is a sentence.",
        "This is another sentence.",
        "Last sentence.",
    ]

    span_getter = make_span_context_getter(
        span_getter=["ents"],
        context_words=0,
        context_sents=2,
        overlap_policy="merge",
    )
    spans = span_getter(doc)
    assert [s.text for s in spans] == [
        (
            "This is a sentence. This is another sentence. "
            "This is a third one. Last sentence."
        )
    ]
