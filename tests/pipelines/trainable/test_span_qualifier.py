import pytest
from pytest import fixture
from spacy.tokens import Span

import edsnlp
from edsnlp.utils.span_getters import get_spans

if not Span.has_extension("label"):
    Span.set_extension("label", default=None)

if not Span.has_extension("event_type"):
    Span.set_extension("event_type", default=None)

if not Span.has_extension("test_negated"):
    Span.set_extension("test_negated", default=False)


@fixture
def gold():
    blank_nlp = edsnlp.blank("eds")
    doc1 = blank_nlp.make_doc("Arret du ttt si folfox inefficace")

    doc1.spans["sc"] = [  # drug = "folfox"
        Span(doc1, 4, 5, "drug"),  # event = "Arret"
        Span(doc1, 0, 1, "event"),  # criteria = "si"
        Span(doc1, 3, 4, "criteria"),
    ]
    doc1.spans["sc"][0]._.test_negated = False
    doc1.spans["sc"][1]._.test_negated = True
    doc1.spans["sc"][2]._.test_negated = False
    doc1.spans["sc"][1]._.event_type = "stop"

    doc1.spans["sent"] = [Span(doc1, 0, 6, "sent")]

    doc2 = blank_nlp.make_doc("Début du traitement")

    span = Span(doc2, 0, 1, "event")
    doc2.ents = [  # drug = "Début"
        span,
    ]
    span._.test_negated = False
    span._.event_type = "start"

    doc2.spans["sent"] = [Span(doc2, 0, 3, "sent")]

    return [doc1, doc2]


@pytest.mark.parametrize("with_constraints_and_not_none", [True, False])
def test_span_qualifier(gold, with_constraints_and_not_none):
    import torch

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.transformer",
        name="transformer",
        config=dict(
            model="prajjwal1/bert-tiny",
            window=128,
            stride=96,
        ),
    )
    nlp.add_pipe(
        "eds.span_qualifier",
        name="qualifier",
        config={
            "embedding": {
                "@factory": "eds.span_pooler",
                "embedding": nlp.get_pipe("transformer"),
                "span_getter": ["ents", "sc"],
            },
            "qualifiers": {"_.event_type": ("event",), "_.test_negated": True}
            if with_constraints_and_not_none
            else ["_.test_negated", "_.event_type"],
            "keep_none": not with_constraints_and_not_none,
        },
    )
    qlf = nlp.get_pipe("qualifier")
    qlf.post_init(gold, set())
    if with_constraints_and_not_none:
        assert qlf.qualifiers == {"_.event_type": ["event"], "_.test_negated": True}
    else:
        assert qlf.qualifiers == {"_.event_type": True, "_.test_negated": True}
    if with_constraints_and_not_none:
        qlf.classifier.bias.data += torch.tensor([0, 1000, 1000, 0])
        assert qlf.bindings == [
            ("_.test_negated", False),
            ("_.test_negated", True),
            ("_.event_type", "start"),
            ("_.event_type", "stop"),
        ]
    else:
        qlf.classifier.bias.data += torch.tensor([0, 1000, 0, 1000, 0])
        assert qlf.bindings == [
            ("_.test_negated", False),
            ("_.test_negated", True),
            ("_.event_type", None),
            ("_.event_type", "start"),
            ("_.event_type", "stop"),
        ]

    pred = qlf.pipe([doc.copy() for doc in gold])
    for doc in pred:
        for ent in get_spans(doc, qlf.embedding.span_getter):
            assert ent._.test_negated is True
            if ent.label_ == "event":
                if with_constraints_and_not_none is not None:
                    assert ent._.event_type == "start"
                else:
                    assert ent._.event_type is None
