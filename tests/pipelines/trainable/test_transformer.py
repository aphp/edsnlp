from pytest import fixture
from spacy.tokens import Span

import edsnlp
from edsnlp.pipelines.trainable.embeddings.transformer.transformer import Transformer
from edsnlp.pipelines.trainable.span_qualifier.span_qualifier import (
    TrainableSpanQualifier,
)

if not Span.has_extension("label"):
    Span.set_extension("label", default=None)

if not Span.has_extension("event_type"):
    Span.set_extension("event_type", default=None)

if not Span.has_extension("test_negated"):
    Span.set_extension("test_negated", default=False)


@fixture
def gold():
    blank_nlp = edsnlp.blank("eds")
    doc1 = blank_nlp.make_doc("Arret du ttt si folfox inefficace. Une autre phrase.")

    doc1.spans["sc"] = [
        Span(doc1, 4, 5, "drug"),  # "folfox"
        Span(doc1, 0, 1, "event"),  # "Arret"
        Span(doc1, 3, 4, "criteria"),  # "si"
    ]
    doc1.spans["sc"][0]._.test_negated = False
    doc1.spans["sc"][1]._.test_negated = True
    doc1.spans["sc"][2]._.test_negated = False
    doc1.spans["sc"][1]._.event_type = "stop"
    doc1.spans["to_embed"] = [doc1[0:5], doc1[7:11]]

    doc1.spans["sent"] = [Span(doc1, 0, 6, "sent")]

    return [doc1]


def test_span_getter(gold):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.transformer",
        name="transformer",
        config=dict(
            model="prajjwal1/bert-tiny", window=128, stride=96, span_getter="to_embed"
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
            "qualifiers": ["_.test_negated", "_.event_type"],
        },
    )
    trf: Transformer = nlp.get_pipe("transformer")
    qlf: TrainableSpanQualifier = nlp.get_pipe("qualifier")
    qlf.post_init(gold, set())
    prep = qlf.make_batch([doc.copy() for doc in gold], supervision=True)
    batch = qlf.collate(prep)
    input_ids = batch["embedding"]["embedding"]["input_ids"]
    mask = batch["embedding"]["embedding"]["input_mask"]
    tok = trf.tokenizer
    assert len(input_ids) == 2
    assert tok.decode(input_ids[0][mask[0]]) == "[CLS] arret du ttt si folfox [SEP]"
    assert tok.decode(input_ids[1][mask[1]]) == "[CLS] une autre phrase. [SEP]"
