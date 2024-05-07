import pytest
from spacy.tokens import Span

import edsnlp
from edsnlp.scorers.ner import NerExactScorer, NerOverlapScorer, NerTokenScorer
from edsnlp.scorers.span_attributes import SpanAttributeScorer


@pytest.fixture(scope="session")
def gold_and_pred():
    nlp = edsnlp.blank("eds")

    gold_doc1 = nlp.make_doc("Le patient a le covid 19.")
    gold_doc1.ents = [
        Span(gold_doc1, 4, 6, label="covid"),  # le covid
    ]
    pred_doc1 = nlp.make_doc("Le patient a le covid 19.")
    pred_doc1.ents = [
        Span(pred_doc1, 4, 6, label="covid"),  # le covid
    ]
    gold_doc2 = nlp.make_doc(
        "Corona: positif. Le cvid est une maladie très très grave."
    )
    gold_doc2.ents = [
        Span(gold_doc2, 0, 1, label="covid"),  # Corona
        Span(gold_doc2, 5, 6, label="covid"),  # cvid
        Span(gold_doc2, 8, 12, label="disease"),  # maladie très très grave
    ]

    pred_doc2 = nlp.make_doc(
        "Corona: positif. Le cvid est une maladie très très grave."
    )
    pred_doc2.ents = [
        Span(pred_doc2, 0, 2, label="covid"),  # Corona:
        Span(pred_doc2, 8, 9, label="disease"),  # maladie
    ]

    return [gold_doc1, gold_doc2], [pred_doc1, pred_doc2]


def test_exact_ner_scorer(gold_and_pred):
    scorer = NerExactScorer("ents", filter_expr="'vid' in doc.text")
    ner_exact_score = scorer(*gold_and_pred)
    assert ner_exact_score["micro"] == {
        "f": 0.2857142857142857,
        "p": 0.3333333333333333,
        "positives": 3,
        "r": 0.25,
        "support": 4,
        "tp": 1,
    }


def test_token_ner_scorer(gold_and_pred):
    scorer = NerTokenScorer("ents", filter_expr="'vid' in doc.text")
    ner_exact_score = scorer(*gold_and_pred)
    assert ner_exact_score["micro"] == {
        "f": 0.6153846153846154,
        "p": 0.8,
        "positives": 5,
        "r": 0.5,
        "support": 8,
        "tp": 4,
    }


def test_overlap_ner_scorer_any(gold_and_pred):
    scorer = NerOverlapScorer(
        "ents", threshold=0.00001, filter_expr="'vid' in doc.text"
    )
    # pred entities: [le covid, Corona:, maladie] => 3
    # gold entities: [le covid, Corona, cvid, maladie très très grave] => 4
    # tp: [le covid, Corona, maladie] => 3
    ner_exact_score = scorer(*gold_and_pred)
    assert ner_exact_score["micro"] == {
        "f": 0.8571428571428572,
        "p": 1,
        "positives": 3,
        "r": 0.75,
        "support": 4,
        "tp": 3,
    }


def test_overlap_ner_scorer_half(gold_and_pred):
    scorer = NerOverlapScorer(
        "ents",
        threshold=0.5,
        filter_expr="'vid' in doc.text",
    )
    # pred entities: [le covid, Corona:, maladie] => 3
    # gold entities: [le covid, Corona, cvid, maladie très très grave] => 4
    # tp: [le covid, Corona] => 2
    ner_exact_score = scorer(*gold_and_pred)
    assert ner_exact_score["micro"] == {
        "f": 0.5714285714285714,
        "p": 0.6666666666666666,
        "positives": 3,
        "r": 0.5,
        "support": 4,
        "tp": 2,
    }


def test_overlap_ner_scorer_full(gold_and_pred):
    scorer = NerOverlapScorer(
        "ents",
        threshold=1.0,
        filter_expr="'vid' in doc.text",
    )
    ner_exact_score = scorer(*gold_and_pred)
    # pred entities: [le covid, Corona:, maladie] => 3
    # gold entities: [le covid, Corona, cvid, maladie très très grave] => 4
    # tp: [le covid] => 2
    assert ner_exact_score["micro"] == {
        "f": 0.2857142857142857,
        "p": 0.3333333333333333,
        "positives": 3,
        "r": 0.25,
        "support": 4,
        "tp": 1,
    }


def test_span_attributes_scorer():
    if not Span.has_extension("negation"):
        Span.set_extension("negation", default=False)
    pred = edsnlp.blank("eds")("Le patient n'a pas le covid 19.")
    gold = edsnlp.blank("eds")("Le patient n'a pas le covid 19.")
    scorer = SpanAttributeScorer(
        "entities",
        "negation",
        default_values={"negation": False},
        filter_expr="'vid' in doc.text",
    )
    pred.spans["entities"] = [
        pred[1:2],
        pred[3:4],
    ]
    pred.spans["entities"][0]._.negation = True
    pred.spans["entities"][1]._.negation = True
    gold.spans["entities"] = [
        gold[1:2],
        gold[3:4],
    ]
    gold.spans["entities"][0]._.negation = False
    gold.spans["entities"][1]._.negation = True
    result = scorer([gold], [pred])
    assert result["micro"] == {
        "ap": 0.5,
        "p": 0.5,
        "r": 1,
        "f": 2 / 3,
        "support": 1,
        "positives": 2,
        "tp": 1,
    }
