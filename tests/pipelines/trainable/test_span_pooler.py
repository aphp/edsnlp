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
    assert word_embeddings.shape == (20, 768)

    assert out.shape == (5, 768)

    # The standalone whitespace wordpiece is not aligned to a word
    # item_indices: [0, 2, 3, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18]
    #                -  ----  ---------------  --  --------------
    # span_offsets: [0, 1,    3,               8,  9]
    # span_indices: [0, 1, 1, 2, 2, 2,  2,  2,  3,  4,  4,  4,  4]

    assert torch.allclose(out[0], word_embeddings[0])
    assert torch.allclose(out[1], word_embeddings[2:4].mean(0))
    assert torch.allclose(out[2], word_embeddings[7:12].mean(0))
    assert torch.allclose(out[3], word_embeddings[14])
    assert torch.allclose(out[4], word_embeddings[15:19].mean(0))


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


@pytest.mark.parametrize("mode", ["mean", "sum", "max", "attention"])
def test_pooler_cnn_layout_and_gradients(mode):
    nlp = edsnlp.blank("eds")
    docs = [nlp.make_doc("One two three four five."), nlp.make_doc("Six.")]
    spans = [[docs[0][1:4], docs[0][2:5], docs[0][3:3]], [docs[1][:]]]
    pooler = eds.span_pooler(
        embedding=eds.text_cnn(embedding=DummyEmbeddings(dim=4), kernel_sizes=(3,)),
        pooling_mode=mode,
    )
    pooler.eval()
    if mode == "attention":
        with torch.no_grad():
            pooler.attention_scorer.weight.fill_(1000)
    prep = [
        pooler.preprocess(doc, spans=selected) for doc, selected in zip(docs, spans)
    ]
    batch = pooler.collate(decompress_dict(list(batch_compress_dict(prep))))
    embeddings = pooler.embedding(batch["embedding"])["embeddings"]
    expected = []
    for row, selected in zip(embeddings.as_tensor(), spans):
        for span in selected:
            values = row[span.start : span.end]
            if not len(span):
                expected.append(row.sum(0) * 0)
            elif mode == "attention":
                expected.append(
                    (values * pooler.attention_scorer(values).softmax(0)).sum(0)
                )
            else:
                expected.append(
                    values.max(0).values if mode == "max" else getattr(values, mode)(0)
                )
    expected = torch.stack(expected)
    actual = pooler(batch)["embeddings"].as_tensor()
    assert torch.allclose(actual, expected, atol=1e-6)
    params = tuple(pooler.parameters())
    actual_grad = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), params)
    for actual, expected in zip(actual_grad, expected_grad):
        assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("score_gap", [0, 40, 1000])
def test_attention_output_and_input_gradients(monkeypatch, score_gap):
    # Compare overlapping wordpiece spans against independent per-span softmax
    nlp = edsnlp.blank("eds")
    docs = [nlp.make_doc("abcdefgh two three four five"), nlp.make_doc("six seven")]
    spans = [[docs[0][1:4], docs[0][2:3], docs[0][3:5], docs[0][3:3]], [docs[1][:]]]
    pooler = eds.span_pooler(
        embedding=DummyEmbeddings(dim=4, word_pooling_mode=False),
        pooling_mode="attention",
    )
    prep = [
        pooler.preprocess(doc, spans=selected) for doc, selected in zip(docs, spans)
    ]
    batch = pooler.collate(decompress_dict(list(batch_compress_dict(prep))))
    embedded = pooler.embedding(batch["embedding"])["embeddings"]
    values = torch.randn(embedded.shape, generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        pooler.attention_scorer.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        values[0, 0, 0] = score_gap
    values.requires_grad_()
    monkeypatch.setattr(
        pooler.embedding,
        "forward",
        lambda batch: {"embeddings": embedded.with_data(values)},
    )
    expected = []
    for row, doc, selected in zip(values.double(), docs, spans):
        offsets = [0]
        for word in doc:
            offsets.append(offsets[-1] + len(word.text[::4]))
        for span in selected:
            tokens = row[offsets[span.start] : offsets[span.end]]
            scores = tokens @ pooler.attention_scorer.weight.double().T
            expected.append((tokens * scores.softmax(0)).sum(0))
    expected = torch.stack(expected)
    actual = pooler(batch)["embeddings"].as_tensor()
    torch.testing.assert_close(actual.double(), expected, atol=1e-5, rtol=1e-4)
    params = (values, pooler.attention_scorer.weight)
    actual_grad = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), params)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
