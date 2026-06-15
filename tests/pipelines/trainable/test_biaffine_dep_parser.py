# ruff:noqa:E402
import numpy as np
import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

pytestmark = pytest.mark.ml

torch = pytest.importorskip("torch")

from edsnlp.pipes.trainable.biaffine_dep_parser.biaffine_dep_parser import (
    TrainableBiaffineDependencyParser,
    chuliu_edmonds,
    chuliu_edmonds_one_root,
)


def test_chuliu_edmonds_allows_multiple_roots():
    scores = np.array(
        [
            [0.0, -10.0, -10.0],
            [5.0, 0.0, 1.0],
            [4.0, 1.0, 0.0],
        ]
    )

    assert chuliu_edmonds(scores).tolist() == [0, 0, 0]
    assert chuliu_edmonds_one_root(scores).tolist() == [0, 0, 1]
    assert scores[1, 0] == 5.0


def test_biaffine_dep_parser_postprocess_assigns_root_arcs():
    doc = Doc(Vocab(), words=["a", "b"])
    parser = TrainableBiaffineDependencyParser.__new__(
        TrainableBiaffineDependencyParser
    )
    parser.decoding_mode = "mst"
    parser.labels = ["root"]

    arc_logits = torch.full((1, 3, 3), -10.0)
    arc_logits[0, 0, 0] = 0.0
    arc_logits[0, 1, 0] = 5.0
    arc_logits[0, 2, 0] = 4.0
    results = {
        "arc_logits": arc_logits,
        "arc_labels": torch.zeros((1, 3, 3), dtype=torch.long),
    }

    parser.postprocess([doc], results, [{"$contexts": [doc[:]]}])

    assert doc[0].head == doc[0]
    assert doc[0].dep_ == "root"
    assert doc[1].head == doc[1]
    assert doc[1].dep_ == "root"
