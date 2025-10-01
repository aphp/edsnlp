import pytest
from confit.utils.random import set_seed
from pytest import fixture
from spacy.tokens import Span

import edsnlp
import edsnlp.pipes as eds
from edsnlp.data.converters import MarkupToDocConverter
from edsnlp.utils.collections import batch_compress_dict, decompress_dict

if not Span.has_extension("label"):
    Span.set_extension("label", default=None)

if not Span.has_extension("event_type"):
    Span.set_extension("event_type", default=None)

if not Span.has_extension("test_negated"):
    Span.set_extension("test_negated", default=False)

pytest.importorskip("torch.nn")


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
    assert res["embeddings"].shape == (9, 128)


def test_transformer_pooling():
    nlp = edsnlp.blank("eds")
    converter = MarkupToDocConverter(tokenizer=nlp.tokenizer)
    doc1 = converter("These are small sentencesstuff.")
    doc2 = converter("A tiny one.")

    def run_trf(word_pooling_mode):
        set_seed(42)
        trf = eds.transformer(
            model="prajjwal1/bert-tiny",
            window=128,
            stride=96,
            word_pooling_mode=word_pooling_mode,
        )
        prep1 = trf.preprocess(doc1)
        prep2 = trf.preprocess(doc2)
        assert prep1["input_ids"] == [
            [2122, 2024, 2235, 11746, 3367, 16093, 2546, 1012]
        ]
        assert prep2["input_ids"] == [[1037, 4714, 2028, 1012]]
        batch = decompress_dict(list(batch_compress_dict([prep1, prep2])))
        batch = trf.collate(batch)
        return trf(batch)

    res_pool = run_trf(word_pooling_mode="mean")
    assert res_pool["embeddings"].shape == (9, 128)

    res_flat = run_trf(word_pooling_mode=False)
    assert res_flat["embeddings"].shape == (12, 128)

    # The second sequence is identical in both cases (only one subword per word)
    # so the last 4 word/subword embeddings should be identical
    assert res_pool["embeddings"][-4:].allclose(res_flat["embeddings"][-4:])
