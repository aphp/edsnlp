import pytest
from spacy.tokens import Span

import edsnlp
from edsnlp.scorers.ner import NerExactScorer, NerTokenScorer
from edsnlp.scorers.span_attributes import SpanAttributeScorer


@pytest.fixture(scope="session")
def gold_and_pred():
    nlp = edsnlp.blank("eds")

    gold_doc1 = nlp.make_doc("Le patient a le covid 19.")
    gold_doc1.ents = [Span(gold_doc1, 4, 6, label="covid")]
    gold_doc2 = nlp.make_doc("Corona: positif. Le cvid est une maladie.")
    gold_doc2.ents = [
        Span(gold_doc2, 0, 1, label="covid"),
        Span(gold_doc2, 5, 6, label="covid"),
    ]

    pred_doc1 = nlp.make_doc("Le patient a le covid 19.")
    pred_doc1.ents = [Span(pred_doc1, 4, 6, label="covid")]
    pred_doc2 = nlp.make_doc("Corona: positif. Le cvid est une maladie.")
    pred_doc2.ents = [Span(pred_doc2, 0, 2, label="covid")]

    return [gold_doc1, gold_doc2], [pred_doc1, pred_doc2]


def test_exact_ner_scorer(gold_and_pred):
    scorer = NerExactScorer("ents", filter_expr="'vid' in doc.text")
    ner_exact_score = scorer(*gold_and_pred)
    assert ner_exact_score["micro"] == {
        "p": 0.5,
        "r": 1 / 3,
        "f": 0.4,
        "support": 3,
        "positives": 2,
        "tp": 1,
    }


def test_token_ner_scorer(gold_and_pred):
    scorer = NerTokenScorer("ents", filter_expr="'vid' in doc.text")
    ner_exact_score = scorer(*gold_and_pred)
    assert ner_exact_score["micro"] == {
        "f": 0.75,
        "p": 0.75,
        "r": 0.75,
        "support": 4,
        "positives": 4,
        "tp": 3,
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
