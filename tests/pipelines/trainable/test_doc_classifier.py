import pytest

import edsnlp
import edsnlp.pipes as eds

pytestmark = pytest.mark.ml

torch = pytest.importorskip("torch")

from edsnlp.pipes.trainable.doc_classifier.heads import (  # noqa: E402
    MultiLabelHead,
    SingleLabelHead,
)


def make_nlp(pooling_mode="mean", **kwargs):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.doc_classifier(
            embedding=eds.doc_pooler(
                pooling_mode=pooling_mode,
                embedding=eds.transformer(
                    model="hf-internal-testing/tiny-random-bert",
                    window=128,
                    stride=96,
                ),
            ),
            **kwargs,
        ),
        name="doc_classifier",
    )
    return nlp


def make_gold(nlp, values):
    docs = []
    for i, attrs in enumerate(values):
        doc = nlp.make_doc(f"Compte rendu numéro {i}.")
        for attr, value in attrs.items():
            setattr(doc._, attr, value)
        docs.append(doc)
    return docs


@pytest.mark.parametrize("pooling_mode", ["mean", "max", "sum", "cls", "attention"])
def test_doc_classifier(pooling_mode):
    nlp = make_nlp(
        pooling_mode,
        heads={
            "dp": SingleLabelHead(labels=["A", "B", "C"]),
            "das": MultiLabelHead(labels=["X", "Y", "Z"], threshold=0.5),
        },
        store_probs=True,
        probs_topk=2,
    )
    doc = nlp("Compte rendu d'hospitalisation.")

    assert doc._.dp in {"A", "B", "C"}
    assert set(doc._.das) <= {"X", "Y", "Z"}
    assert set(doc._.prob) == {"dp", "das"}
    dp_probs = list(doc._.prob["dp"].values())
    assert len(dp_probs) == 2 and dp_probs == sorted(dp_probs, reverse=True)
    assert next(iter(doc._.prob["dp"])) == doc._.dp


def test_training_with_partial_supervision():
    """Documents with and without a gold value for a head can share a batch."""
    nlp = make_nlp(
        heads={
            "dp": SingleLabelHead(),
            "das": MultiLabelHead(class_weights={"X": 1, "Y": 3, "Z": 1}),
        }
    )
    clf = nlp.pipes.doc_classifier
    gold = make_gold(
        nlp,
        [
            {"dp": "A", "das": ["X", "Z"]},
            {"dp": "B", "das": None},
            {"dp": None, "das": ["Y"]},
        ],
    )
    clf.post_init(gold, set())
    assert clf.heads["dp"].label2id == {"A": 0, "B": 1}
    assert clf.heads["das"].label2id == {"X": 0, "Y": 1, "Z": 2}

    batch = clf.prepare_batch(gold, supervision=True)
    assert batch["targets"]["dp"].tolist() == [0, 1, -100]
    assert clf.heads["das"].is_supervised(batch["targets"]["das"]).tolist() == [
        True,
        False,
        True,
    ]
    loss = clf(batch)["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()

    # A batch that supervises no document of a head simply skips it
    batch = clf.prepare_batch(gold[1:2], supervision=True)
    assert torch.isfinite(clf(batch)["loss"])


def test_config_and_serialization(tmp_path):
    from confit import Config

    nlp = edsnlp.load(
        Config.from_str(
            """
            [nlp]
            lang = "eds"
            pipeline = ["doc_classifier"]

            [nlp.components.doc_classifier]
            @factory = "eds.doc_classifier"
            store_probs = true
            probs_threshold = 0.0

            [nlp.components.doc_classifier.embedding]
            @factory = "eds.doc_pooler"
            pooling_mode = "mean"

            [nlp.components.doc_classifier.embedding.embedding]
            @factory = "eds.transformer"
            model = "hf-internal-testing/tiny-random-bert"
            window = 128
            stride = 96

            [nlp.components.doc_classifier.heads.dp]
            @misc = "eds.single_label_head"
            labels = ["A", "B"]
            hidden_size = 8
            activation_mode = "gelu"
            dropout_rate = 0.1
            layer_norm = true

            [nlp.components.doc_classifier.heads.das]
            @misc = "eds.multi_label_head"
            threshold = 0.3
            """
        )
    )
    nlp.pipes.doc_classifier.post_init(
        make_gold(nlp, [{"dp": "A", "das": ["X"]}, {"dp": "B", "das": ["Y"]}]),
        set(),
    )
    nlp.train(False)
    before = nlp("Compte rendu d'hospitalisation.")
    assert len(before._.prob["das"]) == 2

    nlp.to_disk(tmp_path / "model")
    reloaded = edsnlp.load(tmp_path / "model")

    head = reloaded.pipes.doc_classifier.heads["das"]
    assert head.threshold == 0.3
    assert head.label2id == {"X": 0, "Y": 1}
    after = reloaded("Compte rendu d'hospitalisation.")
    assert (after._.dp, after._.das) == (before._.dp, before._.das)

    # A config edited before reloading, e.g. to fine-tune, takes precedence
    config_path = tmp_path / "model" / "config.cfg"
    config_path.write_text(
        config_path.read_text().replace("threshold = 0.3", "threshold = 0.7")
    )
    assert edsnlp.load(tmp_path / "model").pipes.doc_classifier.heads[
        "das"
    ].threshold == pytest.approx(0.7)
