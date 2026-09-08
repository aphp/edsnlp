import pytest
from dummy_embeddings import DummyEmbeddings
from spacy.tokens import Span

import edsnlp
import edsnlp.pipes as eds

pytestmark = pytest.mark.ml

pytest.importorskip("torch.nn")


def test_relation_detector_ffn_without_inter_span_embedding():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.relation_detector_ffn(
            span_embedding=eds.span_pooler(
                embedding=DummyEmbeddings(dim=4, word_pooling_mode=False),
                pooling_mode="mean",
            ),
            candidate_getter=[
                {
                    "head": {"ents": ["drug"]},
                    "tail": {"ents": ["problem"]},
                    "labels": ["treats"],
                    "symmetric": True,
                }
            ],
            hidden_size=4,
        ),
        name="relations",
    )
    detector = nlp.get_pipe("relations")

    doc = nlp.make_doc("aspirin pain")
    head = Span(doc, 0, 1, "drug")
    tail = Span(doc, 1, 2, "problem")
    doc.ents = [head, tail]
    head._.rel["treats"] = [tail]

    batch = detector.prepare_batch([doc], supervision=True)
    result = detector.module_forward(batch)

    assert batch["rel_head_idx"].tolist() == [0]
    assert batch["rel_tail_idx"].tolist() == [1]
    # Convert the scalar tensor explicitly so this also works with torch
    # versions that do not allow a tensor in an assertion.
    assert result["loss"].item() > 0

    detector.classifier.bias.data.fill_(1)
    predictions = detector.module_forward(detector.prepare_batch([doc]))
    detector.postprocess([doc], predictions, [detector.preprocess(doc)])

    assert isinstance(head._.rel["treats"], set)
    assert tail in head._.rel["treats"]
    assert head in tail._.rel["treats"]


def test_relation_detector_ffn_with_inter_span_embedding():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.relation_detector_ffn(
            span_embedding=eds.span_pooler(
                embedding=DummyEmbeddings(dim=4, word_pooling_mode=False),
                pooling_mode="mean",
            ),
            inter_span_embedding=eds.span_pooler(
                embedding=DummyEmbeddings(dim=4, word_pooling_mode=False),
                pooling_mode="mean",
            ),
            candidate_getter=[
                {
                    "head": {"ents": ["drug"]},
                    "tail": {"ents": ["problem"]},
                    "labels": ["treats"],
                    "symmetric": True,
                }
            ],
            hidden_size=4,
        ),
        name="relations",
    )
    detector = nlp.get_pipe("relations")

    doc = nlp.make_doc("aspirin for pain")
    head = Span(doc, 0, 1, "drug")
    tail = Span(doc, 2, 3, "problem")
    doc.ents = [head, tail]
    head._.rel["treats"] = [tail]

    batch = detector.prepare_batch([doc], supervision=True)
    result = detector.module_forward(batch)

    assert batch["rel_head_idx"].tolist() == [0]
    assert batch["rel_tail_idx"].tolist() == [1]
    assert batch["inter_embedding"] is not None
    assert result["loss"].item() > 0

    detector.classifier.bias.data.fill_(1)
    predictions = detector.module_forward(detector.prepare_batch([doc]))
    detector.postprocess([doc], predictions, [detector.preprocess(doc)])

    assert isinstance(head._.rel["treats"], set)
    assert tail in head._.rel["treats"]
    assert head in tail._.rel["treats"]
