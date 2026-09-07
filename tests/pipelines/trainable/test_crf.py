import pytest

pytestmark = pytest.mark.ml

pytest.importorskip("torch.nn")


@pytest.mark.parametrize("layout", ["contiguous", "strided", "folded"])
def test_bioul_spans(layout):
    import torch

    from edsnlp.pipes.trainable.layers.crf import MultiLabelBIOULDecoder

    tags = (
        torch.tensor(
            [
                [
                    [2, 1, 3, 0, 4, 0, 0, 0],
                    [0, 0, 2, 3, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [3, 1, 4, 2, 1, 0, 3, 0],
                    [4, 4, 0, 0, 0, 0, 0, 2],
                    [0, 1, 3, 0, 2, 2, 3, 0],
                ],
            ]
        )
        .transpose(1, 2)
        .contiguous()
    )
    if layout == "strided":
        storage = torch.zeros((2, 16, 3), dtype=torch.long)
        storage[:, ::2] = tags
        tags = storage[:, ::2]
    elif layout == "folded":
        import foldedtensor as ft

        tags = ft.as_folded_tensor(
            [[0] * 8, [0] * 8],
            full_names=("context", "word"),
            data_dims=("context", "word"),
        ).with_data(tags)
    before = tags.clone()
    expected = torch.tensor(
        [
            [0, 0, 0, 3],
            [0, 0, 4, 5],
            [0, 1, 2, 4],
            [0, 1, 5, 7],
            [1, 0, 0, 1],
            [1, 0, 1, 2],
            [1, 0, 2, 3],
            [1, 0, 3, 5],
            [1, 0, 6, 7],
            [1, 1, 0, 1],
            [1, 1, 1, 2],
            [1, 1, 7, 8],
            [1, 2, 1, 3],
            [1, 2, 4, 5],
            [1, 2, 5, 7],
        ]
    )
    actual = MultiLabelBIOULDecoder.tags_to_spans(tags)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(tags, before)


@pytest.mark.parametrize(
    "shape", [(0, 5, 3), (2, 0, 3), (2, 5, 0), (0, 0, 0), (2, 5, 3)]
)
def test_bioul_no_spans(shape):
    import torch

    from edsnlp.pipes.trainable.layers.crf import MultiLabelBIOULDecoder

    actual = MultiLabelBIOULDecoder.tags_to_spans(torch.zeros(shape, dtype=torch.long))
    torch.testing.assert_close(actual, torch.empty((0, 4), dtype=torch.long))
