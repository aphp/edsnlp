import abc
from collections import defaultdict
from typing import Any, Dict, Optional

from edsnlp import registry
from edsnlp.scorers import make_examples, prf
from edsnlp.utils.span_getters import SpanGetter, SpanGetterArg, get_spans


def ner_exact_scorer(
    *args,
    span_getter: SpanGetter,
    micro_key: str = "micro",
    filter_expr: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Scores the extracted entities that may be overlapping or nested
    by looking in the spans returned by a given SpanGetter object.

    Parameters
    ----------
    *args: Examples
        The examples to score, either a tuple of (golds, preds) or a list of
        spacy.training.Example objects
    span_getter: SpanGetter
        The span getter to use to extract the spans from the document
    micro_key: str
        The key to use to store the micro-averaged results for spans of all types
    filter_expr: str
        The filter expression to use to filter the documents

    Returns
    -------
    Dict[str, Any]
    """
    examples = make_examples(*args)
    if filter_expr is not None:
        filter_fn = eval(f"lambda doc: {filter_expr}")
        examples = [eg for eg in examples if filter_fn(eg.reference)]
    labels = defaultdict(lambda: (set(), set()))
    labels["micro"] = (set(), set())
    for eg_idx, eg in enumerate(examples):
        for span in (
            span_getter(eg.predicted)
            if callable(span_getter)
            else get_spans(eg.predicted, span_getter)
        ):
            labels[span.label_][0].add((eg_idx, span.start, span.end, span.label_))
            labels[micro_key][0].add((eg_idx, span.start, span.end, span.label_))

        for span in (
            span_getter(eg.reference)
            if callable(span_getter)
            else get_spans(eg.reference, span_getter)
        ):
            labels[span.label_][1].add((eg_idx, span.start, span.end, span.label_))
            labels[micro_key][1].add((eg_idx, span.start, span.end, span.label_))

    return {name: prf(pred, gold) for name, (pred, gold) in labels.items()}


def ner_token_scorer(
    *args,
    span_getter: SpanGetter,
    micro_key: str = "micro",
    filter_expr: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Scores the extracted entities that may be overlapping or nested
    by looking in `doc.ents`, and `doc.spans`, and comparing the predicted
    and gold entities at the TOKEN level.

    Parameters
    ----------
    *args: Examples
        The examples to score, either a tuple of (golds, preds) or a list of
        spacy.training.Example objects
    span_getter: SpanGetter
        The span getter to use to extract the spans from the document
    micro_key: str
        The key to use to store the micro-averaged results for spans of all types
    filter_expr: str
        The filter expression to use to filter the documents

    Returns
    -------
    Dict[str, Any]
    """
    examples = make_examples(*args)
    if filter_expr is not None:
        filter_fn = eval(f"lambda doc: {filter_expr}")
        examples = [eg for eg in examples if filter_fn(eg.reference)]
    # label -> pred, gold
    labels = defaultdict(lambda: (set(), set()))
    labels["micro"] = (set(), set())
    for eg_idx, eg in enumerate(examples):
        for span in (
            span_getter(eg.predicted)
            if callable(span_getter)
            else get_spans(eg.predicted, span_getter)
        ):
            for i in range(span.start, span.end):
                labels[span.label_][0].add((eg_idx, i, span.label_))
                labels[micro_key][0].add((eg_idx, i, span.label_))

        for span in (
            span_getter(eg.reference)
            if callable(span_getter)
            else get_spans(eg.reference, span_getter)
        ):
            for i in range(span.start, span.end):
                labels[span.label_][1].add((eg_idx, i, span.label_))
                labels[micro_key][1].add((eg_idx, i, span.label_))

    return {name: prf(pred, gold) for name, (pred, gold) in labels.items()}


class NerScorer(abc.ABC):
    span_getter: SpanGetter

    def __call__(self, *examples) -> Dict[str, Any]:
        raise NotImplementedError()


@registry.scorers.register("eds.ner_exact_scorer")
class NerExactScorer(NerScorer):
    def __init__(
        self,
        span_getter: SpanGetterArg,
        micro_key: str = "micro",
        filter_expr: Optional[str] = None,
    ):
        self.span_getter = span_getter
        self.micro_key = micro_key
        self.filter_expr = filter_expr

    def __call__(self, *examples):
        return ner_exact_scorer(
            *examples,
            span_getter=self.span_getter,
            micro_key=self.micro_key,
            filter_expr=self.filter_expr,
        )


@registry.scorers.register("eds.ner_token_scorer")
class NerTokenScorer(NerScorer):
    def __init__(
        self,
        span_getter: SpanGetterArg,
        micro_key: str = "micro",
        filter_expr: Optional[str] = None,
    ):
        self.span_getter = span_getter
        self.micro_key = micro_key
        self.filter_expr = filter_expr

    def __call__(self, *examples):
        return ner_token_scorer(
            *examples,
            span_getter=self.span_getter,
            micro_key=self.micro_key,
            filter_expr=self.filter_expr,
        )


# For backward compatibility
create_ner_exact_scorer = NerExactScorer
create_ner_token_scorer = NerTokenScorer
