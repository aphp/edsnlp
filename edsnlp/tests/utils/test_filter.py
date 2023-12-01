from spacy.tokens import Doc, Span

from edsnlp.utils.filter import filter_spans


def test_filter_spans(doc: Doc):
    spans = [
        doc[0:3],
        doc[0:4],
        doc[1:2],
        doc[0:2],
        doc[0:3],
    ]

    filtered = filter_spans(spans)

    assert len(filtered) == 1
    assert len(filtered[0]) == 4


def test_filter_spans_strict_nesting(doc: Doc):
    spans = [
        doc[0:5],
        doc[1:4],
    ]

    filtered = filter_spans(spans)

    assert len(filtered) == 1
    assert len(filtered[0]) == 5


def test_label_to_remove(doc: Doc):

    spans = [
        Span(doc, 0, 5, label="test"),
        Span(doc, 6, 10, label="test"),
        Span(doc, 6, 10, label="remove"),
    ]

    filtered = filter_spans(spans, label_to_remove="remove")

    assert len(filtered) == 2

    spans = [
        Span(doc, 6, 10, label="remove"),
        Span(doc, 0, 5, label="test"),
        Span(doc, 6, 10, label="test"),
    ]

    filtered = filter_spans(spans, label_to_remove="remove")

    assert len(filtered) == 1
