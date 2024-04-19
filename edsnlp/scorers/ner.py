import abc
from collections import defaultdict
from typing import Any, Dict, Iterable

from spacy.training import Example

from edsnlp import registry
from edsnlp.scorers import make_examples, prf
from edsnlp.utils.span_getters import SpanGetter, SpanGetterArg, get_spans


def ner_exact_scorer(
    examples: Iterable[Example],
    span_getter: SpanGetter,
    micro_key: str = "micro",
) -> Dict[str, Any]:
    """
    Scores the extracted entities that may be overlapping or nested
    by looking in the spans returned by a given SpanGetter object.

    Parameters
    ----------
    examples: Iterable[Example]
        The examples to score
    span_getter: SpanGetter
        The span getter to use to extract the spans from the document
    micro_key: str
        The key to use to store the micro-averaged results for spans of all types

    Returns
    -------
    Dict[str, Any]
    """
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
    examples: Iterable[Example],
    span_getter: SpanGetter,
    micro_key: str = "micro",
) -> Dict[str, Any]:
    """
    Scores the extracted entities that may be overlapping or nested
    by looking in `doc.ents`, and `doc.spans`, and comparing the predicted
    and gold entities at the TOKEN level.

    Parameters
    ----------
    examples: Iterable[Example]
        The examples to score
    span_getter: SpanGetter
        The span getter to use to extract the spans from the document
    micro_key: str
        The key to use to store the micro-averaged results for spans of all types

    Returns
    -------
    Dict[str, Any]
    """
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

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()


@registry.scorers.register("eds.ner_exact_scorer")
class NerExactScorer(NerScorer):
    def __init__(self, span_getter: SpanGetterArg):
        self.span_getter = span_getter

    def __call__(self, *args, **kwargs):
        return ner_exact_scorer(make_examples(*args, **kwargs), self.span_getter)


@registry.scorers.register("eds.ner_token_scorer")
class NerTokenScorer(NerScorer):
    def __init__(self, span_getter: SpanGetterArg):
        self.span_getter = span_getter

    def __call__(self, *args, **kwargs):
        return ner_token_scorer(make_examples(*args, **kwargs), self.span_getter)


# For backward compatibility
create_ner_exact_scorer = NerExactScorer
create_ner_token_scorer = NerTokenScorer
