import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.ml


def test_bioul_overlap_decodes_crossing_spans():
    from edsnlp.pipes.trainable.layers.crf import MultiLabelBIOULDecoder

    tags = torch.tensor([[[2], [1], [2], [1], [1], [3], [1], [3]]])
    decoder = MultiLabelBIOULDecoder(
        num_labels=1,
        learnable_transitions=False,
        allow_overlap=True,
    )

    assert torch.equal(
        decoder.tags_to_spans(tags),
        torch.tensor(
            [
                [0, 0, 6, 0],
                [0, 0, 8, 0],
                [0, 2, 6, 0],
                [0, 2, 8, 0],
            ]
        ),
    )
    assert not decoder.forbidden_transitions[tags[0, :-1, 0], tags[0, 1:, 0]].any()


def test_bioul_overlap_is_disabled_by_default():
    from edsnlp.pipes.trainable.layers.crf import MultiLabelBIOULDecoder

    default = MultiLabelBIOULDecoder(num_labels=2)
    explicit = MultiLabelBIOULDecoder(num_labels=2, allow_overlap=False)

    assert torch.equal(default.forbidden_transitions, explicit.forbidden_transitions)
