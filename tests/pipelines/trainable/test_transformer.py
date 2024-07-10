from pytest import fixture
from spacy.tokens import Span

import edsnlp
from edsnlp.utils.collections import batch_compress_dict, decompress_dict

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
    from edsnlp.pipes.trainable.embeddings.transformer.transformer import Transformer
    from edsnlp.pipes.trainable.span_qualifier.span_qualifier import (
        TrainableSpanQualifier,
    )

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.transformer",
        name="transformer",
        config=dict(
            model="prajjwal1/bert-tiny",
            window=128,
            stride=96,
            quantization=None,
        ),
    )
    nlp.add_pipe(
        "eds.span_qualifier",
        name="qualifier",
        config={
            "embedding": {
                "@factory": "eds.span_pooler",
                "embedding": nlp.get_pipe("transformer"),
            },
            "span_getter": ["ents", "sc"],
            "context_getter": ["to_embed"],
            "qualifiers": ["_.test_negated", "_.event_type"],
        },
    )
    trf: Transformer = nlp.get_pipe("transformer")
    qlf: TrainableSpanQualifier = nlp.get_pipe("qualifier")
    qlf.post_init(gold, set())
    batch = qlf.prepare_batch([doc.copy() for doc in gold], supervision=True)
    input_ids = batch["embedding"]["embedding"]["input_ids"]
    mask = input_ids.mask
    tok = trf.tokenizer
    assert len(input_ids) == 2
    assert tok.decode(input_ids[0][mask[0]]) == "[CLS] arret du ttt si folfox [SEP]"
    assert tok.decode(input_ids[1][mask[1]]) == "[CLS] une autre phrase. [SEP]"

    # Transformer alone with prompts (usually passed by the caller component)
    prep = trf.preprocess(
        gold[0],
        contexts=gold[0].spans["to_embed"],
        prompts=["Extract the drugs", "Extract the drugs"],
    )
    batch = decompress_dict(list(batch_compress_dict([prep])))
    batch = trf.collate(batch)
    batch = trf.batch_to_device(batch, device=trf.device)
    res = trf(batch)
    assert res["embeddings"].shape == (2, 5, 128)
