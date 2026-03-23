import pytest
from spacy.tokens import Span

import edsnlp
from edsnlp.data.converters import MarkupToDocConverter
from edsnlp.metrics import average_precision
from edsnlp.metrics.ner import NerExactScorer, NerOverlapScorer, NerTokenScorer
from edsnlp.metrics.span_attribute import SpanAttributeMetric, SpanAttributeScorer


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


def test_span_attribute_metric_split_by_values():
    Span.set_extension("status", default=None, force=True)

    pred = edsnlp.blank("eds")("a b c")
    gold = edsnlp.blank("eds")("a b c")

    pred.spans["entities"] = [pred[0:1], pred[1:2], pred[2:3]]
    gold.spans["entities"] = [gold[0:1], gold[1:2], gold[2:3]]

    pred.spans["entities"][0]._.status = "present"
    pred.spans["entities"][1]._.status = "absent"
    pred.spans["entities"][2]._.status = "present"

    gold.spans["entities"][0]._.status = "present"
    gold.spans["entities"][1]._.status = "absent"
    gold.spans["entities"][2]._.status = "absent"

    metric = SpanAttributeMetric(
        span_getter="entities",
        attributes=["status"],
        split_by_values="status",
    )
    result = metric([gold], [pred])

    assert set(result) == {"micro", "status"}
    assert set(result["status"]) == {"micro", "absent", "present"}

    assert result["micro"]["tp"] == 2
    assert result["micro"]["positives"] == 3
    assert result["micro"]["support"] == 3
    assert result["micro"]["p"] == pytest.approx(2 / 3)
    assert result["micro"]["r"] == pytest.approx(2 / 3)
    assert result["micro"]["f"] == pytest.approx(2 / 3)

    assert result["status"]["micro"]["tp"] == 2
    assert result["status"]["micro"]["positives"] == 3
    assert result["status"]["micro"]["support"] == 3
    assert result["status"]["micro"]["p"] == pytest.approx(2 / 3)
    assert result["status"]["micro"]["r"] == pytest.approx(2 / 3)
    assert result["status"]["micro"]["f"] == pytest.approx(2 / 3)

    assert result["status"]["present"]["tp"] == 1
    assert result["status"]["present"]["positives"] == 2
    assert result["status"]["present"]["support"] == 1
    assert result["status"]["present"]["p"] == pytest.approx(0.5)
    assert result["status"]["present"]["r"] == pytest.approx(1.0)
    assert result["status"]["present"]["f"] == pytest.approx(2 / 3)

    assert result["status"]["absent"]["tp"] == 1
    assert result["status"]["absent"]["positives"] == 1
    assert result["status"]["absent"]["support"] == 2
    assert result["status"]["absent"]["p"] == pytest.approx(1.0)
    assert result["status"]["absent"]["r"] == pytest.approx(0.5)
    assert result["status"]["absent"]["f"] == pytest.approx(2 / 3)


def test_span_attribute_metric_self_comparison_uses_assigned_value_prob():
    """
    See: https://github.com/aphp/edsnlp/issues/428
    And regardless of the argmax val, we use the prob of the assigned value
    for the AP computation
    """
    Span.set_extension("status", default=None, force=True)
    Span.set_extension("prob", default={}, force=True)

    conv = MarkupToDocConverter(preset="md", span_setter="entities")
    doc = conv("[a](ENT status=present prob={'status':{'absent':0.9,'present':0.1}}) b")

    metric = SpanAttributeMetric(
        span_getter="entities",
        attributes=["status"],
    )
    result = metric([doc], [doc])

    assert result["micro"]["p"] == 1.0
    assert result["micro"]["r"] == 1.0
    assert result["micro"]["f"] == 1.0
    assert result["micro"]["ap"] == 1.0

    assert result["status"]["p"] == 1.0
    assert result["status"]["r"] == 1.0
    assert result["status"]["f"] == 1.0
    assert result["status"]["ap"] == 1.0
