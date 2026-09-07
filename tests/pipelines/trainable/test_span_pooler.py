import pytest

import edsnlp
from edsnlp.utils.collections import batch_compress_dict, decompress_dict

pytestmark = pytest.mark.ml

pytest.importorskip("torch.nn")


@pytest.fixture
def embedding():
    import foldedtensor as ft
    import torch

    from edsnlp.pipes.trainable.embeddings.typing import WordEmbeddingComponent

    # Use numbered token text as embedding row indices for pooling comparisons
    class TokenEmbedding(WordEmbeddingComponent):
        output_size = 3

        def __init__(self, nlp=None):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(39, dtype=torch.float).reshape(13, 3)
            )

        def preprocess(self, doc, *, contexts, **kwargs):
            return {"words": [[int(w.text) for w in ctx] for ctx in contexts]}

        def collate(self, batch):
            return {
                "words": ft.as_folded_tensor(
                    batch["words"],
                    full_names=("sample", "context", "word"),
                    data_dims=("context", "word"),
                    dtype=torch.long,
                )
            }

        def forward(self, batch):
            words = batch["words"]
            return {"embeddings": words.with_data(self.weight[words.as_tensor()])}

    return TokenEmbedding()


# Compare pooled values and gradients across contexts and overlapping or empty spans
@pytest.mark.parametrize("mode", ["mean", "sum", "max"])
@pytest.mark.parametrize("hidden_size", [None, 2])
@pytest.mark.parametrize(
    "bounds",
    [
        [[(1, 1), (0, 2), (1, 5), (6, 9)], [(1, 3), (3, 3)]],
        [[(0, 3), (2, 5)], []],
        [[], []],
    ],
)
def test_span_pooler(embedding, mode, hidden_size, bounds):
    import torch

    from edsnlp.pipes.trainable.embeddings.span_pooler.span_pooler import SpanPooler

    nlp = edsnlp.blank("eds")
    docs = [nlp.make_doc("0 1 2 3 4 5 6 7 8"), nlp.make_doc("9 10 11 12")]
    # Use two embedding contexts in the first document and one in the second
    contexts = [[docs[0][:5], docs[0][5:]], [docs[1][:]]]
    pooler = SpanPooler(embedding=embedding, pooling_mode=mode, hidden_size=hidden_size)
    inputs = [
        pooler.preprocess(
            doc,
            spans=[doc[begin:end] for begin, end in spans],
            contexts=ctx,
        )
        for doc, ctx, spans in zip(docs, contexts, bounds)
    ]
    batch = pooler.collate(decompress_dict(list(batch_compress_dict(inputs))))
    actual = pooler(batch)["embeddings"]
    assert actual.lengths == [[2], [len(spans) for spans in bounds]]
    assert actual.shape == (sum(map(len, bounds)), pooler.output_size)
    if not any(bounds):
        return

    expected = []
    for doc, spans in zip(docs, bounds):
        for begin, end in spans:
            words = embedding.weight[[int(token.text) for token in doc[begin:end]]]
            if begin == end:
                expected.append(words.new_zeros(embedding.output_size))
            elif mode == "max":
                expected.append(words.max(0).values)
            else:
                expected.append(getattr(words, mode)(0))
    expected = pooler.projector(torch.stack(expected))
    torch.testing.assert_close(actual.as_tensor(), expected)
    actual_grads = torch.autograd.grad(actual.square().sum(), pooler.parameters())
    expected_grads = torch.autograd.grad(expected.square().sum(), pooler.parameters())
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
