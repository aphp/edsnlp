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
        context_words=2,
    )
    assert [span_getter(s).text for s in doc.ents] == [
        "This is a sentence. This",
        "This is another sentence. This",
        ". Last sentence.",
    ]

    span_getter = make_span_context_getter(
        context_words=2,
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
        "This is a sentence. This is another sentence. This is a third one.",
        "This is a sentence. This is another sentence. This is a third one. Last "
        "sentence.",
        "This is another sentence. This is a third one. Last sentence.",
    ]
