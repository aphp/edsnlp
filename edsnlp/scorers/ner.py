from typing import Any, Dict, Iterable

from spacy.training import Example

from edsnlp import registry
from edsnlp.pipelines.base import SpanGetter, SpanGetterArg, get_spans
from edsnlp.scorers import make_examples


def ner_exact_scorer(
    examples: Iterable[Example], span_getter: SpanGetter
) -> Dict[str, Any]:
    """
    Scores the extracted entities that may be overlapping or nested
    by looking in the spans returned by a given SpanGetter object.

    Parameters
    ----------
    examples: Iterable[Example]
    span_getter: SpanGetter

    Returns
    -------
    Dict[str, Any]
    """
    pred_spans = set()
    gold_spans = set()
    for eg_idx, eg in enumerate(examples):
        for span in (
            span_getter(eg.predicted)
            if callable(span_getter)
            else get_spans(eg.predicted, span_getter)
        ):
            pred_spans.add((eg_idx, span.start, span.end, span.label_))

        for span in (
            span_getter(eg.reference)
            if callable(span_getter)
            else get_spans(eg.reference, span_getter)
        ):
            gold_spans.add((eg_idx, span.start, span.end, span.label_))

    tp = len(pred_spans & gold_spans)

    return {
        "ents_p": tp / len(pred_spans) if pred_spans else float(len(gold_spans) == 0),
        "ents_r": tp / len(gold_spans) if gold_spans else float(len(gold_spans) == 0),
        "ents_f": 2 * tp / (len(pred_spans) + len(gold_spans))
        if pred_spans or gold_spans
        else float(len(pred_spans) == len(gold_spans)),
        "support": len(gold_spans),
    }


def ner_token_scorer(
    examples: Iterable[Example], span_getter: SpanGetter
) -> Dict[str, Any]:
    """
    Scores the extracted entities that may be overlapping or nested
    by looking in `doc.ents`, and `doc.spans`, and comparing the predicted
    and gold entities at the TOKEN level.

    Parameters
    ----------
    examples: Iterable[Example]
    span_getter: SpanGetter

    Returns
    -------
    Dict[str, Any]
    """
    pred_spans = set()
    gold_spans = set()
    for eg_idx, eg in enumerate(examples):
        for span in (
            span_getter(eg.predicted)
            if callable(span_getter)
            else get_spans(eg.predicted, span_getter)
        ):
            for i in range(span.start, span.end):
                pred_spans.add((eg_idx, i, span.label_))

        for span in (
            span_getter(eg.reference)
            if callable(span_getter)
            else get_spans(eg.reference, span_getter)
        ):
            for i in range(span.start, span.end):
                gold_spans.add((eg_idx, i, span.label_))

    tp = len(pred_spans & gold_spans)

    return {
        "ents_p": tp / len(pred_spans) if pred_spans else float(tp == len(pred_spans)),
        "ents_r": tp / len(gold_spans) if gold_spans else float(tp == len(gold_spans)),
        "ents_f": 2 * tp / (len(pred_spans) + len(gold_spans))
        if pred_spans or gold_spans
        else float(len(pred_spans) == len(gold_spans)),
        "support": len(gold_spans),
    }


@registry.scorers.register("eds.ner_exact_scorer")
def create_ner_exact_scorer(
    span_getter: SpanGetterArg,
):
    return lambda *args, **kwargs: ner_exact_scorer(
        make_examples(*args, **kwargs), span_getter
    )


@registry.scorers.register("eds.ner_token_scorer")
def create_ner_token_scorer(
    span_getter: SpanGetterArg,
):
    return lambda *args: ner_token_scorer(make_examples(*args), span_getter)
