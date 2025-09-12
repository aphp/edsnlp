import confit.utils.random
import pytest
from dummy_embeddings import DummyEmbeddings

import edsnlp
import edsnlp.pipes as eds
from edsnlp.data.converters import MarkupToDocConverter
from edsnlp.pipes.trainable.embeddings.span_pooler.span_pooler import SpanPooler
from edsnlp.utils.collections import batch_compress_dict, decompress_dict

pytest.importorskip("torch.nn")

import torch


@pytest.mark.parametrize(
    "word_pooling_mode,shape",
    [
        ("mean", (2, 5, 2)),
        (False, (2, 6, 2)),
    ],
)
def test_dummy_embeddings(word_pooling_mode, shape):
    confit.utils.random.set_seed(42)
    converter = MarkupToDocConverter()
    doc1 = converter("This is a sentence.")
    doc2 = converter("A shorter one.")
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        DummyEmbeddings(dim=2, word_pooling_mode=word_pooling_mode), name="embeddings"
    )
    embedder: DummyEmbeddings = nlp.pipes.embeddings

    prep1 = embedder.preprocess(doc1)
    prep2 = embedder.preprocess(doc2)
    pivoted_prep = decompress_dict(list(batch_compress_dict([prep1, prep2])))
    batch = embedder.collate(pivoted_prep)
    out = embedder.forward(batch)["embeddings"]

    assert out.shape == shape


@pytest.mark.parametrize("span_pooling_mode", ["max", "mean", "attention"])
def test_span_pooler_on_words(span_pooling_mode):
    confit.utils.random.set_seed(42)
    converter = MarkupToDocConverter()
    doc1 = converter("[This](ent) is [a sentence](ent). This is [small one](ent).")
    doc2 = converter("An [even shorter one](ent) !")
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.span_pooler(
            embedding=DummyEmbeddings(dim=2),
            pooling_mode=span_pooling_mode,
        )
    )
    pooler: SpanPooler = nlp.pipes.span_pooler

    prep1 = pooler.preprocess(doc1, spans=doc1.ents)
    prep2 = pooler.preprocess(doc2, spans=doc2.ents)
    pivoted_prep = decompress_dict(list(batch_compress_dict([prep1, prep2])))
    batch = pooler.collate(pivoted_prep)
    out = pooler.forward(batch)["embeddings"]

    assert out.shape == (4, 2)
    out = out.refold("sample", "span")

    assert out.shape == (2, 3, 2)
    if span_pooling_mode == "attention":
        expected = [
            [[0.0000, 0.0000], [3.8102, 3.8102], [9.7554, 9.7554]],
            [[3.6865, 3.6865], [0.0000, 0.0000], [0.0000, 0.0000]],
        ]
    elif span_pooling_mode == "mean":
        expected = [
            [[0.0000, 0.0000], [3.0000, 3.0000], [9.5000, 9.5000]],
            [[2.6667, 2.6667], [0.0000, 0.0000], [0.0000, 0.0000]],
        ]
    elif span_pooling_mode == "max":
        expected = [
            [[0.0000, 0.0000], [4.0000, 4.0000], [10.0000, 10.0000]],
            [[4.0000, 4.0000], [0.0000, 0.0000], [0.0000, 0.0000]],
        ]
    else:
        raise ValueError(f"Unknown pooling mode: {span_pooling_mode}")
    assert torch.allclose(out, torch.tensor(expected), atol=1e-4)


@pytest.mark.parametrize("span_pooling_mode", ["max", "mean", "attention"])
def test_span_pooler_on_tokens(span_pooling_mode):
    confit.utils.random.set_seed(42)
    converter = MarkupToDocConverter()
    doc1 = converter("[This](ent) is [a sentence](ent). This is [small one](ent).")
    doc2 = converter("An [even shorter one](ent) !")
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.span_pooler(
            embedding=DummyEmbeddings(dim=2, word_pooling_mode=False),
            pooling_mode=span_pooling_mode,
        )
    )
    pooler: SpanPooler = nlp.pipes.span_pooler

    prep1 = pooler.preprocess(doc1, spans=doc1.ents)
    prep2 = pooler.preprocess(doc2, spans=doc2.ents)
    pivoted_prep = decompress_dict(list(batch_compress_dict([prep1, prep2])))
    batch = pooler.collate(pivoted_prep)
    out = pooler.forward(batch)["embeddings"]

    assert out.shape == (4, 2)
    out = out.refold("sample", "span")

    assert out.shape == (2, 3, 2)
    if span_pooling_mode == "attention":
        expected = [
            [[0.0000, 0.0000], [3.6265, 3.6265], [9.6265, 9.6265]],
            [[3.5655, 3.5655], [0.0000, 0.0000], [0.0000, 0.0000]],
        ]
    elif span_pooling_mode == "mean":
        expected = [
            [[0.0000, 0.0000], [3.0000, 3.0000], [9.0000, 9.0000]],
            [[2.5000, 2.5000], [0.0000, 0.0000], [0.0000, 0.0000]],
        ]
    elif span_pooling_mode == "max":
        expected = [
            [[0.0000, 0.0000], [4.0000, 4.0000], [10.0000, 10.0000]],
            [[4.0000, 4.0000], [0.0000, 0.0000], [0.0000, 0.0000]],
        ]
    else:
        raise ValueError(f"Unknown pooling mode: {span_pooling_mode}")
    assert torch.allclose(out, torch.tensor(expected), atol=1e-4)


def test_span_pooler_on_flat_hf_tokens():
    confit.utils.random.set_seed(42)
    converter = MarkupToDocConverter()
    doc1 = converter("[This](ent) is [a sentence](ent). This is [small one](ent).")
    doc2 = converter("An [even](ent) [shorter one](ent) !")
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.span_pooler(
            embedding=eds.transformer(
                model="almanach/camembert-base",
                word_pooling_mode=False,
            ),
            pooling_mode="mean",
        )
    )
    pooler: SpanPooler = nlp.pipes.span_pooler

    prep1 = pooler.preprocess(doc1, spans=doc1.ents)
    prep2 = pooler.preprocess(doc2, spans=doc2.ents)
    pivoted_prep = decompress_dict(list(batch_compress_dict([prep1, prep2])))
    print(
        nlp.pipes.span_pooler.embedding.tokenizer.convert_ids_to_tokens(
            prep2["embedding"]["input_ids"][0]
        )
    )
    # fmt: off
    assert prep1["embedding"]["input_ids"] == [
        [
            17526,  # ▁This: 0  -> span 0
            2856,  #  ▁is: 1
            33,  #    ▁a: 2 -> span 1
            22625,  # ▁sentence: 3 -> span 1
            9,  #     .: 4
            17526,  # ▁This: 5
            2856,  #  ▁is: 6
            52,  #    ▁s: 7 -> span 2
            215,  #   m: 8 -> span 2
            3645,  #  all: 9 -> span 2
            91,  #    ▁on: 10 -> span 2
            35,  #    e: 11 -> span 2
            9,  #     .: 12
        ],
    ]
    # '▁An', '▁', 'even', '▁short', 'er', '▁on', 'e', '▁!'
    assert prep2["embedding"]["input_ids"] == [
        [
            2764,  #  ▁An: 13
            21,  #    ▁: 14
            15999,  # even: 15 -> span 3
            9161,  #  short: 16 -> span 4
            108,  #   er: 17 -> span 4
            91,  #    ▁on: 18 -> span 4
            35,  #    e: 19 -> span 4
            83,  #    ▁!: 20
        ]
    ]
    # fmt: on
    batch = pooler.collate(pivoted_prep)
    out = pooler.forward(batch)["embeddings"]

    word_embeddings = pooler.embedding(batch["embedding"])["embeddings"]
    assert word_embeddings.shape == (21, 768)

    assert out.shape == (5, 768)

    # item_indices: [0, 2, 3, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
    #                -  ----  ---------------  ------  --------------
    # span_offsets: [0, 1,    3,               8,      10]
    # span_indices: [0, 1, 1, 2, 2, 2, 2,  2,  3,  3,  4,  4,  4,  4]

    assert torch.allclose(out[0], word_embeddings[0])
    assert torch.allclose(out[1], word_embeddings[2:4].mean(0))
    assert torch.allclose(out[2], word_embeddings[7:12].mean(0))
    assert torch.allclose(out[3], word_embeddings[14:16].mean(0))
    assert torch.allclose(out[4], word_embeddings[16:20].mean(0))


def test_span_pooler_on_pooled_hf_tokens():
    confit.utils.random.set_seed(42)
    converter = MarkupToDocConverter()
    doc1 = converter("[This](ent) is [a sentence](ent). This is [small one](ent).")
    doc2 = converter("An [even](ent) [shorter one](ent) !")
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.span_pooler(
            embedding=eds.transformer(
                model="almanach/camembert-base",
                word_pooling_mode="mean",
            ),
            pooling_mode="mean",
        )
    )
    pooler: SpanPooler = nlp.pipes.span_pooler

    prep1 = pooler.preprocess(doc1, spans=doc1.ents)
    prep2 = pooler.preprocess(doc2, spans=doc2.ents)
    pivoted_prep = decompress_dict(list(batch_compress_dict([prep1, prep2])))
    print(
        nlp.pipes.span_pooler.embedding.tokenizer.convert_ids_to_tokens(
            prep2["embedding"]["input_ids"][0]
        )
    )
    # fmt: off
    assert prep1["embedding"]["input_ids"] == [
        [
            17526,  #          ▁This: 0  -> span 0
            2856,  #           ▁is: 1
            33,  #             ▁a: 2 -> span 1
            22625,  #          ▁sentence: 3 -> span 1
            9,  #              .: 4
            17526,  #          ▁This: 5
            2856,  #           ▁is: 6
            52, 215, 3645,  #  ▁s m all: 7 -> span 2
            91, 35,  #         ▁on e: 8 -> span 2
            9,  #              .: 9
        ],
    ]
    # '▁An', '▁', 'even', '▁short', 'er', '▁on', 'e', '▁!'
    assert prep2["embedding"]["input_ids"] == [
        [
            2764,  #  ▁An: 10
            21, 15999,  #    ▁, even: 11 -> span 3
            9161, 108, #  short er: 12 -> span 4
            91, 35,  #    ▁on e: 13 -> span 4
            83,  #    ▁!: 14
        ]
    ]
    # fmt: on
    batch = pooler.collate(pivoted_prep)
    out = pooler.forward(batch)["embeddings"]

    word_embeddings = pooler.embedding(batch["embedding"])["embeddings"]
    assert word_embeddings.shape == (15, 768)

    assert out.shape == (5, 768)

    # item_indices: [0, 2, 3, 7, 8, 11, 12, 13]
    #                -  ----  ----  --  ------
    # span_offsets: [0, 1,    3,    5,  6     ]

    assert torch.allclose(out[0], word_embeddings[0])
    assert torch.allclose(out[1], word_embeddings[2:4].mean(0))
    assert torch.allclose(out[2], word_embeddings[7:9].mean(0))
    assert torch.allclose(out[3], word_embeddings[11])
    assert torch.allclose(out[4], word_embeddings[12:14].mean(0))
