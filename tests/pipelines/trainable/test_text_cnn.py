import foldedtensor as ft
import pytest
from dummy_embeddings import DummyEmbeddings

import edsnlp.pipes as eds

pytest.importorskip("torch")
import torch
import torch.nn.functional as F


@pytest.mark.parametrize(
    "dims",
    [("word",), ("context", "word"), ("sample", "word"), ("sample", "context", "word")],
)
@pytest.mark.parametrize(
    "kernels,normalize,output_size",
    [((1,), "pre", 4), ((2,), "post", 4), ((3, 4, 5), "none", 4), ((3, 4, 5), None, 6)],
)
def test_cnn_layouts_and_gradients(monkeypatch, dims, kernels, normalize, output_size):
    torch.manual_seed(0)
    cnn = eds.text_cnn(
        embedding=DummyEmbeddings(dim=4),
        kernel_sizes=kernels,
        normalize=normalize or "none",
        residual=normalize is not None,
        output_size=output_size,
    )
    for data in ([[[0], [0, 0], []], [], [[0, 0, 0]]], [[[]], [], [[]]]):
        structure = ft.as_folded_tensor(
            data, full_names=("sample", "context", "word"), data_dims=("word",)
        )
        words = torch.randn(structure.numel(), 4, requires_grad=True)
        embedding = structure.with_data(words).refold(dims)
        monkeypatch.setattr(
            cnn.embedding, "collate", lambda batch: {"out_structure": embedding.lengths}
        )
        monkeypatch.setattr(
            cnn.embedding, "forward", lambda batch: {"embeddings": embedding}
        )
        batch = cnn.collate({"embedding": {}})
        actual = cnn(batch)["embeddings"]
        assert actual.full_names == embedding.full_names
        assert actual.lengths == embedding.lengths
        assert actual.data_dims == batch["out_structure"].data_dims == (2,)
        assert actual.shape == (len(words), output_size)
        if not len(words):
            actual.sum().backward()
            assert words.grad is not None
            continue
        expected = []
        for seq in words.split(structure.lengths["word"]):
            if len(seq):
                convoluted = torch.cat(
                    [
                        conv(
                            F.pad(
                                seq.T.unsqueeze(0),
                                (
                                    conv.kernel_size[0] // 2,
                                    (conv.kernel_size[0] - 1) // 2,
                                ),
                            )
                        )[0].T
                        for conv in cnn.module.convolutions
                    ],
                    dim=-1,
                )
                result = cnn.module.linear(torch.relu(convoluted))
                expected.append(
                    cnn.module.residual(seq, result)
                    if cnn.module.residual is not None
                    else result
                )
        expected = torch.cat(expected)
        assert torch.allclose(actual.as_tensor(), expected, atol=1e-5, rtol=1e-4)
        params = (words, *cnn.parameters())
        actual_grad = torch.autograd.grad(
            actual.square().sum(), params, retain_graph=True
        )
        expected_grad = torch.autograd.grad(expected.square().sum(), params)
        for actual, expected in zip(actual_grad, expected_grad):
            assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
