import pytest
from spacy.tokens import Span

from edsnlp.data.converters import MarkupToDocConverter
from edsnlp.metrics import average_precision
from edsnlp.metrics.ner import NerExactScorer, NerOverlapScorer, NerTokenScorer
from edsnlp.metrics.span_attribute import SpanAttributeScorer


@pytest.fixture(scope="session")
def gold_and_pred():
    conv = MarkupToDocConverter(preset="md")
    gold_doc1 = conv("Le patient a [le covid](covid) 19.")
    pred_doc1 = conv("Le patient a [le covid](covid) 19.")
    gold_doc2 = conv(
        "[Corona](covid): positif. Le [cvid](covid) est "
        "une [maladie très très grave](disease)."
    )
    pred_doc2 = conv(
        "[Corona:](covid) positif. Le cvid est une [maladie](disease) très très grave."
    )

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
    conv = MarkupToDocConverter(preset="md", span_setter="entities")
    pred = conv(
        "Le patient n'a pas [le covid](ENT negation=true) "
        "[aujourd'hui](ENT negation=true)."
    )
    gold = conv(
        "Le patient n'a pas [le covid](ENT negation=false) "
        "[aujourd'hui](ENT negation=true)."
    )
    scorer = SpanAttributeScorer(
        "entities",
        "negation",
        default_values={"negation": False},
        filter_expr="'vid' in doc.text",
        split_by_values="negation",
    )
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
    assert result["negation"]["micro"] == {
        "ap": 0.5,
        "p": 0.5,
        "r": 1,
        "f": 2 / 3,
        "support": 1,
        "positives": 2,
        "tp": 1,
    }
    assert result["negation"]["True"] == {
        "ap": 0.5,
        "p": 0.5,
        "r": 1,
        "f": 2 / 3,
        "support": 1,
        "positives": 2,
        "tp": 1,
    }


def test_average_precision():
    assert average_precision({"tp": 1.0}, {"tp"}) == 1.0
    assert average_precision({"tp": 1.0, "fp": 0.0}, {"tp"}) == 1.0
    assert average_precision({"fp": 1.0, "tp": 0.0}, {"tp"}) == 0.5
    assert average_precision(
        {"tp1": 1.0, "fp": 0.5, "tp2": 0.0}, {"tp1", "tp2"}
    ) == pytest.approx(5 / 6)
