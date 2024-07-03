import pytest
import spacy
from pytest import fixture
from spacy.tokens import Span
from typing_extensions import Literal

import edsnlp
import edsnlp.pipes as eds

if not Span.has_extension("cui"):
    Span.set_extension("cui", default=None)


@fixture
def gold():
    blank_nlp = edsnlp.blank("eds")
    doc1 = blank_nlp.make_doc("Prise de folfox puis doliprane par le patient.")

    doc1.ents = [
        Span(doc1, 0, 1, "event"),
        Span(doc1, 2, 3, "drug"),
        Span(doc1, 4, 5, "drug"),
        Span(doc1, 7, 8, "livb"),
    ]
    doc1.ents[0]._.cui = None
    doc1.ents[1]._.cui = "CONCEPT1"
    doc1.ents[2]._.cui = "CONCEPT2"
    doc1.ents[3]._.cui = "CONCEPT3"

    return [doc1]


@pytest.mark.parametrize(
    "metric,probability_mode,reference_mode",
    [
        ("cosine", "softmax", "concept"),
        ("cosine", "sigmoid", "concept"),
        ("dot", "sigmoid", "synonym"),
    ],
)
def test_span_linker(
    metric: Literal["cosine", "dot"],
    probability_mode: Literal["softmax", "sigmoid"],
    reference_mode: Literal["concept", "synonym"],
    gold,
    tmp_path,
):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.span_linker(
            rescale=20.0,
            threshold=0.8,
            metric=metric,
            reference_mode=reference_mode,
            probability_mode=probability_mode,
            # just to maximize coverage, prefer init_weights=True in practice
            init_weights=True if reference_mode == "concept" else False,
            span_getter=["ents"],
            context_getter=["ents"],
            embedding=eds.span_pooler(
                hidden_size=128,
                embedding=eds.transformer(
                    model="prajjwal1/bert-tiny",
                    window=128,
                    stride=96,
                ),
            ),
        ),
        name="linker",
    )

    def convert(entry):
        doc = nlp.make_doc(entry["STR"].lower()[:100])
        span = spacy.tokens.Span(
            doc,
            0,
            len(doc),
            label=entry["GRP"],
        )
        span._.cui = entry["CUI"]
        doc.ents = [span]
        return doc

    linker: eds.span_linker = nlp.get_pipe("linker")

    synonyms = edsnlp.data.from_iterable(
        [
            {"STR": "folfox", "CUI": "CONCEPT1", "GRP": "drug"},
            {"STR": "doliprane", "CUI": "CONCEPT2", "GRP": "drug"},
            {"STR": "doliprone", "CUI": "CONCEPT2", "GRP": "drug"},
            {"STR": "patiente", "CUI": "CONCEPT3", "GRP": "livb"},
            {"STR": "docteur", "CUI": "CONCEPT4", "GRP": "livb"},
        ],
        converter=convert,
    )
    assert linker.attributes == ["cui"]
    nlp.post_init(synonyms)
    pred = nlp.pipe([doc.copy() for doc in gold])

    pred_cuis = [span._.cui for doc in pred for span in doc.ents]
    gold_cuis = [span._.cui for doc in pred for span in doc.ents]

    assert pred_cuis == gold_cuis

    if probability_mode == "softmax":
        batch = linker.prepare_batch([doc.copy() for doc in gold], supervision=True)
        results = linker.forward(batch)
        assert results["loss"] > 0

        nlp.to_disk(tmp_path / "model")
        nlp = edsnlp.load(tmp_path / "model")

    assert "TrainableSpanLinker" in str(linker)
