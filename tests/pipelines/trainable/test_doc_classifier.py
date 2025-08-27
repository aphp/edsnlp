import pytest

import edsnlp
import edsnlp.pipes as eds

pytest.importorskip("torch.nn")


@pytest.mark.parametrize("pooling_mode", ["mean", "max", "cls", "sum"])
@pytest.mark.parametrize("label_attr", ["label", "alive"])
@pytest.mark.parametrize("num_classes", [2, 10])
def test_doc_classifier(pooling_mode, label_attr, num_classes):
    nlp = edsnlp.blank("eds")
    doc = nlp.make_doc("Le patient est mort.")

    nlp.add_pipe(
        eds.doc_classifier(
            embedding=eds.doc_pooler(
                pooling_mode=pooling_mode,
                embedding=eds.transformer(
                    model="prajjwal1/bert-tiny",
                    window=128,
                    stride=96,
                ),
            ),
            num_classes=num_classes,
            label_attr=label_attr,
        ),
        name="doc_classifier",
    )
    doc = nlp(doc)
    label = getattr(doc._, label_attr, None)
    assert label in range(0, num_classes)
