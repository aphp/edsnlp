import pytest

import edsnlp
import edsnlp.pipes as eds

pytestmark = pytest.mark.ml

pytest.importorskip("torch.nn")

from edsnlp.pipes.trainable.doc_classifier.heads import (  # noqa: E402
    MultiLabelHead,
    SingleLabelHead,
)


def _pooler(pooling_mode):
    return eds.doc_pooler(
        pooling_mode=pooling_mode,
        embedding=eds.transformer(
            model="hf-internal-testing/tiny-random-bert",
            window=128,
            stride=96,
        ),
    )


@pytest.mark.parametrize("pooling_mode", ["mean", "max", "sum", "cls", "attention"])
def test_single_label_head(pooling_mode):
    labels = ["alive", "dead"]
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.doc_classifier(
            embedding=_pooler(pooling_mode),
            heads={"status": SingleLabelHead(labels=labels, loss="ce")},
        ),
        name="doc_classifier",
    )
    doc = nlp("Le patient est mort.")
    assert doc._.status in labels


def test_multi_head_with_count_topk():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.doc_classifier(
            embedding=_pooler("mean"),
            heads={
                "dp": SingleLabelHead(labels=["A", "B", "C"], loss="ce"),
                "das": MultiLabelHead(
                    labels=["X", "Y", "Z", "W"],
                    loss="bce",
                    selection="topk",
                    count_head="das_count",
                ),
                "das_count": SingleLabelHead(labels=[0, 1, 2, 3], loss="ce"),
            },
        ),
        name="doc_classifier",
    )
    doc = nlp("Compte rendu d'hospitalisation.")
    assert doc._.dp in {"A", "B", "C"}
    assert isinstance(doc._.das, list)
    # topk selects exactly `das_count` labels
    assert len(doc._.das) == int(doc._.das_count)
    assert all(code in {"X", "Y", "Z", "W"} for code in doc._.das)
    # alternative (threshold) decoding is exposed for comparison
    assert isinstance(doc._.das_alt, list)


def test_multi_label_threshold():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.doc_classifier(
            embedding=_pooler("mean"),
            heads={
                "das": MultiLabelHead(
                    labels=["X", "Y", "Z"],
                    loss="bce",
                    selection="threshold",
                    threshold=0.5,
                ),
            },
        ),
        name="doc_classifier",
    )
    doc = nlp("Texte clinique.")
    assert isinstance(doc._.das, list)
    assert all(code in {"X", "Y", "Z"} for code in doc._.das)


def _gold_docs(nlp, values):
    """Build gold documents, setting `doc._.<head>` from the given mappings."""
    docs = []
    for i, attrs in enumerate(values):
        doc = nlp.make_doc(f"Compte rendu numéro {i}.")
        for attr, value in attrs.items():
            setattr(doc._, attr, value)
        docs.append(doc)
    return docs


def _classifier(**heads):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.doc_classifier(embedding=_pooler("mean"), heads=heads),
        name="doc_classifier",
    )
    return nlp


def test_training_step_and_partial_supervision():
    """A document without a gold value for a head must not supervise it."""
    import torch

    nlp = _classifier(
        dp=SingleLabelHead(labels=["A", "B"], loss="ce"),
        das=MultiLabelHead(labels=["X", "Y", "Z"], loss="bce"),
    )
    clf = nlp.get_pipe("doc_classifier")

    fully_labelled = _gold_docs(nlp, [{"dp": "A", "das": ["X", "Z"]}])
    dp_only = _gold_docs(nlp, [{"dp": "B", "das": None}])

    # Multi-hot target has exactly one 1 per gold label.
    target = clf.heads["das"].build_target(fully_labelled[0], "das")
    assert target.tolist() == [1.0, 0.0, 1.0]
    assert clf.heads["das"].build_target(dp_only[0], "das") is None

    batch = clf.prepare_batch(fully_labelled, supervision=True)
    assert set(batch["targets"]) == {"dp", "das"}
    loss = clf(batch)["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()

    # The DP-only stream trains without error, and only through the `dp` head.
    batch = clf.prepare_batch(dp_only, supervision=True)
    assert set(batch["targets"]) == {"dp"}
    assert torch.isfinite(clf(batch)["loss"])


def test_labels_inferred_from_gold():
    """Heads built without an explicit label list scan the gold data."""
    nlp = _classifier(
        dp=SingleLabelHead(loss="ce"),
        das=MultiLabelHead(loss="bce"),
    )
    clf = nlp.get_pipe("doc_classifier")
    gold = _gold_docs(
        nlp,
        [
            {"dp": "A", "das": ["X", "Y"]},
            {"dp": "B", "das": ["Z"]},
        ],
    )
    clf.post_init(gold, set())

    assert clf.heads["dp"].label2id == {"A": 0, "B": 1}
    assert clf.heads["das"].label2id == {"X": 0, "Y": 1, "Z": 2}
    assert clf.heads["dp"].built and clf.heads["das"].built

    doc = nlp("Compte rendu.")
    assert doc._.dp in {"A", "B"}


@pytest.mark.parametrize("as_str", [True, False])
def test_labels_and_weights_from_pickle(tmp_path, as_str):
    """Labels and class weights can be given as a list/dict or as a path to a pickle."""
    import pickle

    import torch

    labels_path = tmp_path / "labels.pkl"
    with open(labels_path, "wb") as f:
        pickle.dump(["alive", "dead"], f)
    weights_path = tmp_path / "weights.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump({"alive": 90, "dead": 10}, f)

    head = SingleLabelHead(
        labels=str(labels_path) if as_str else labels_path,
        class_weights=str(weights_path) if as_str else weights_path,
        hidden_size=8,
        activation_mode="gelu",
        dropout_rate=0.1,
        layer_norm=True,
        loss="ce",
    )
    assert head.label2id == {"alive": 0, "dead": 1}
    assert head._freq_dict == {"alive": 90, "dead": 10}

    head.build(input_size=4)
    assert head.norm is not None
    # The rare class gets the larger weight.
    assert head.loss_fn.weight[1] > head.loss_fn.weight[0]
    logits = head(torch.zeros(3, 4))
    assert logits.shape == (3, 2)

    head.classifier.bias.data[:] = torch.tensor([0.0, 1000.0])
    assert head.decode(head(torch.zeros(3, 4))) == ["dead", "dead", "dead"]


def test_focal_loss_with_class_weights():
    """The focal head runs and weights rare classes more than frequent ones."""
    import torch

    head = SingleLabelHead(
        labels=["rare", "frequent"],
        class_weights={"rare": 1, "frequent": 99},
        loss="focal",
    )
    head.build(input_size=4)
    weights = head.loss_fn.alpha
    assert weights[head.label2id["rare"]] > weights[head.label2id["frequent"]]

    logits = torch.zeros(2, 2, requires_grad=True)
    loss = head.compute_loss(logits, torch.tensor([0, 1]))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_misconfigurations_are_rejected_upfront():
    """Every misconfiguration below used to fail silently or late."""
    with pytest.raises(ValueError, match="non-empty mapping"):
        eds.doc_classifier(embedding=_pooler("mean"), heads={})

    # An unsupported loss used to be caught only in `build()`, i.e. during
    # `post_init`, once the model and the data were already loaded.
    with pytest.raises(ValueError, match="Unsupported loss 'hinge'"):
        SingleLabelHead(labels=["A"], loss="hinge")
    with pytest.raises(ValueError, match="Unsupported loss 'ce'"):
        MultiLabelHead(labels=["X"], loss="ce")

    # `topk` without a count head used to predict an empty label set forever.
    with pytest.raises(ValueError, match="needs a `count_head`"):
        MultiLabelHead(labels=["X"], loss="bce", selection="topk")

    # So did a count head naming a head that does not exist.
    with pytest.raises(ValueError, match="not one of the heads"):
        _classifier(
            das=MultiLabelHead(
                labels=["X"],
                loss="bce",
                selection="topk",
                count_head="missing",
            )
        )


def test_cls_pooling_requires_a_cls_output():
    """`cls` pooling used to raise a bare `KeyError: 'cls'` at the first forward."""
    import torch

    pooler = eds.doc_pooler(
        pooling_mode="cls",
        embedding=eds.text_cnn(
            embedding=eds.transformer(
                model="hf-internal-testing/tiny-random-bert",
                window=128,
                stride=96,
            )
        ),
    )
    nlp = edsnlp.blank("eds")
    batch = pooler.prepare_batch([nlp("Texte clinique.")], device=torch.device("cpu"))
    with pytest.raises(ValueError, match="requires an embedding that returns"):
        pooler(batch)


def test_heads_are_instantiable_from_a_config():
    """The `heads:` mapping is the documented way to configure the component."""
    from confit import Config

    nlp = edsnlp.load(
        Config.from_str(
            """
            [nlp]
            lang = "eds"
            pipeline = ["doc_classifier"]

            [nlp.components.doc_classifier]
            @factory = "eds.doc_classifier"

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
            loss = "ce"

            [nlp.components.doc_classifier.heads.das]
            @misc = "eds.multi_label_head"
            labels = ["X", "Y", "Z"]
            loss = "bce"
            selection = "topk"
            count_head = "das_count"

            [nlp.components.doc_classifier.heads.das_count]
            @misc = "eds.single_label_head"
            labels = [0, 1, 2]
            loss = "ce"
            """
        )
    )
    doc = nlp("Compte rendu d'hospitalisation.")
    assert doc._.dp in {"A", "B"}
    assert len(doc._.das) == int(doc._.das_count)


def test_to_disk_from_disk_roundtrip(tmp_path):
    """Head configuration, labels and predictions survive a save/load cycle."""
    nlp = _classifier(
        dp=SingleLabelHead(labels=["A", "B", "C"], loss="ce"),
        das=MultiLabelHead(
            labels=["X", "Y", "Z", "W"],
            loss="bce",
            selection="topk",
            threshold=0.3,
            count_head="das_count",
        ),
        das_count=SingleLabelHead(labels=[0, 1, 2, 3], loss="ce"),
    )
    before = nlp("Compte rendu d'hospitalisation.")

    nlp.to_disk(tmp_path / "model")
    reloaded = edsnlp.load(tmp_path / "model")

    head = reloaded.pipes.doc_classifier.heads["das"]
    assert head.selection == "topk"
    assert head.threshold == 0.3
    assert head.count_head == "das_count"
    assert head.label2id == {"X": 0, "Y": 1, "Z": 2, "W": 3}

    after = reloaded("Compte rendu d'hospitalisation.")
    assert after._.dp == before._.dp
    assert after._.das == before._.das
    assert after._.das_count == before._.das_count


@pytest.mark.parametrize(
    "reduction,expected_shape",
    [("mean", ()), ("sum", ()), ("none", (3,))],
)
def test_focal_loss_reductions(reduction, expected_shape):
    import torch

    from edsnlp.pipes.trainable.doc_classifier.heads import FocalLoss

    loss = FocalLoss(reduction=reduction)(torch.randn(3, 2), torch.tensor([0, 1, 0]))
    assert tuple(loss.shape) == expected_shape


def test_gold_scan_skips_unannotated_documents():
    """A document with no value for a head must not contribute to its label set."""
    nlp = _classifier(das=MultiLabelHead(loss="bce"))
    clf = nlp.get_pipe("doc_classifier")
    gold = _gold_docs(nlp, [{"das": ["X", "Y"]}, {"das": None}, {"das": ["Z"]}])

    clf.post_init(gold, set())
    assert clf.heads["das"].label2id == {"X": 0, "Y": 1, "Z": 2}


def test_single_label_targets():
    """Gold values are mapped to class indices; unknown labels are rejected."""
    nlp = _classifier(dp=SingleLabelHead(labels=["A", "B"], loss="ce"))
    head = nlp.get_pipe("doc_classifier").heads["dp"]

    labelled, unlabelled, unknown = _gold_docs(
        nlp, [{"dp": "B"}, {"dp": None}, {"dp": "Z"}]
    )
    assert head.build_target(labelled, "dp").item() == 1
    assert head.build_target(unlabelled, "dp") is None
    with pytest.raises(ValueError, match="not in label2id"):
        head.build_target(unknown, "dp")


def test_multi_label_target_accepts_a_single_string():
    """A gold value that is a bare string counts as a one-element label set."""
    nlp = _classifier(das=MultiLabelHead(labels=["X", "Y", "Z"], loss="bce"))
    head = nlp.get_pipe("doc_classifier").heads["das"]

    (doc,) = _gold_docs(nlp, [{"das": "Y"}])
    assert head.build_target(doc, "das").tolist() == [0.0, 1.0, 0.0]


@pytest.mark.parametrize(
    "head_cls,loss,target",
    [
        (SingleLabelHead, "ce", [0, 1]),
        (MultiLabelHead, "bce", [[1.0, 0.0], [0.0, 1.0]]),
    ],
)
def test_class_weights_follow_the_logits_device(head_cls, loss, target):
    """Weights are built on the CPU and must be moved next to the logits."""
    import torch

    head = head_cls(
        labels=["rare", "frequent"],
        class_weights={"rare": 1, "frequent": 99},
        loss=loss,
    )
    head.build(input_size=4)
    logits = torch.zeros(2, 2, requires_grad=True)
    value = head.compute_loss(logits, torch.tensor(target))
    assert value.ndim == 0 and torch.isfinite(value)


def test_postprocess_is_a_noop_without_logits():
    """In training mode the forward returns no logits, and nothing is decoded."""
    import torch

    nlp = _classifier(dp=SingleLabelHead(labels=["A", "B"], loss="ce"))
    clf = nlp.get_pipe("doc_classifier")
    docs = [nlp.make_doc("Compte rendu.")]

    out = clf.postprocess(docs, {"loss": torch.tensor(0.0), "logits": None}, {})
    assert out is docs
    assert docs[0]._.dp is None


def test_roundtrip_rebuilds_heads_whose_labels_came_from_gold(tmp_path):
    """Heads built at `post_init` are unbuilt on load, and `from_disk` rebuilds them."""
    nlp = _classifier(
        dp=SingleLabelHead(loss="ce"),
        das=MultiLabelHead(loss="bce"),
    )
    clf = nlp.get_pipe("doc_classifier")
    clf.post_init(
        _gold_docs(nlp, [{"dp": "A", "das": ["X"]}, {"dp": "B", "das": ["Y"]}]),
        set(),
    )
    before = nlp("Compte rendu d'hospitalisation.")

    nlp.to_disk(tmp_path / "model")
    reloaded = edsnlp.load(tmp_path / "model")

    heads = reloaded.pipes.doc_classifier.heads
    assert heads["dp"].label2id == {"A": 0, "B": 1}
    assert heads["das"].built
    after = reloaded("Compte rendu d'hospitalisation.")
    assert (after._.dp, after._.das) == (before._.dp, before._.das)


def test_head_cannot_be_built_before_its_labels_are_known():
    with pytest.raises(ValueError, match="before its labels are known"):
        SingleLabelHead(loss="ce").build(input_size=4)
