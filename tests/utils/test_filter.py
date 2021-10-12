from spacy.tokens import Doc

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
