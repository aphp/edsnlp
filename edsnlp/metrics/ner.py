"""
We provide several metrics to evaluate the performance of Named Entity Recognition (NER) components.
Let's look at an example and see how they differ. We'll use the following two documents: a reference
document (ref) and a document with predicted entities (pred).

+-------------------------------------------------------------+------------------------------------------+
| pred                                                        | ref                                      |
+=============================================================+==========================================+
| *La*{.chip data-chip=PER} *patiente*{.chip data-chip=PER} a | La *patiente*{.chip data-chip=PER} a     |
| une *fièvre aigüe*{.chip data-chip=DIS}                     | *une fièvre*{.chip data-chip=DIS} aigüe. |
+-------------------------------------------------------------+------------------------------------------+

Let's create matching documents in EDS-NLP using the following code snippet:

```python
from edsnlp.data.converters import MarkupToDocConverter

conv = MarkupToDocConverter(preset="md", span_setter="entities")

pred = conv("[La](PER) [patiente](PER) a une [fièvre aiguë](DIS).")
ref = conv("La [patiente](PER) a [une fièvre](DIS) aiguë.")
```
"""  # noqa: E501

import abc
from collections import defaultdict
from typing import Any, Dict, Optional

from edsnlp import registry
from edsnlp.metrics import Examples, make_examples, prf
from edsnlp.utils.span_getters import SpanGetter, SpanGetterArg, get_spans


def ner_exact_metric(
    examples: Examples,
    span_getter: SpanGetter,
    micro_key: str = "micro",
    filter_expr: Optional[str] = None,
) -> Dict[str, Any]:
    examples = make_examples(examples)
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


def ner_token_metric(
    examples: Examples,
    span_getter: SpanGetter,
    micro_key: str = "micro",
    filter_expr: Optional[str] = None,
) -> Dict[str, Any]:
    examples = make_examples(examples)
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


def dice(span1, span2):
    """
    Compute the Dice coefficient between two spans
    """
    intersection = max(0, min(span1.end, span2.end) - max(span1.start, span2.start))
    return 2 * intersection / (span1.end - span1.start + span2.end - span2.start)


def ner_overlap_metric(
    examples: Examples,
    span_getter: SpanGetter,
    micro_key: str = "micro",
    filter_expr: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    examples = make_examples(*examples)
    if filter_expr is not None:
        filter_fn = eval(f"lambda doc: {filter_expr}")
        examples = [eg for eg in examples if filter_fn(eg.reference)]
    # label -> pred, gold, matched_pred, matched_gold
    counters = defaultdict(lambda: [0, 0, 0, 0])
    counters["micro"] = [0, 0, 0, 0]
    for eg_idx, eg in enumerate(examples):
        pred_spans = set(
            span_getter(eg.predicted)
            if callable(span_getter)
            else get_spans(eg.predicted, span_getter)
        )
        gold_spans = set(
            span_getter(eg.reference)
            if callable(span_getter)
            else get_spans(eg.reference, span_getter)
        )
        for gold_span in gold_spans:
            counters[gold_span.label_][1] += 1
            counters[micro_key][1] += 1
        for pred_span in pred_spans:
            counters[pred_span.label_][0] += 1
            counters[micro_key][0] += 1

        for pred_span in pred_spans:
            overlaps = [
                (gold_span, dice(pred_span, gold_span))
                for gold_span in gold_spans
                if gold_span.label == pred_span.label
            ]
            matching_gold_span, overlap = max(
                overlaps, key=lambda x: x[1], default=(None, 0)
            )
            if matching_gold_span is not None and overlap >= threshold:
                counters[pred_span.label_][2] += 1
                counters[micro_key][2] += 1

        for gold_span in gold_spans:
            overlaps = [
                (pred_span, dice(gold_span, pred_span)) for pred_span in pred_spans
            ]
            matching_pred_span, overlap = max(
                overlaps, key=lambda x: x[1], default=(None, 0)
            )
            if (
                matching_pred_span is not None
                and overlap >= threshold
                and gold_span.label == matching_pred_span.label
            ):
                counters[gold_span.label_][3] += 1
                counters[micro_key][3] += 1

    results = {}
    for name, (num_pred, num_gold, true_pred, true_gold) in counters.items():
        p = (true_pred / num_pred) if num_pred else 1.0
        r = (true_gold / num_gold) if num_gold else 1.0
        f = (
            2 / (num_pred / true_pred + num_gold / true_gold)
            if true_pred and true_gold
            else 0.0
        )
        results[name] = {
            "f": f,
            "p": p,
            "r": r,
            "tp": true_pred,
            "support": num_gold,  # num gold
            "positives": num_pred,  # num predicted
        }
    return results


class NerMetric(abc.ABC):
    span_getter: SpanGetter

    def __call__(self, *examples) -> Dict[str, Any]:
        raise NotImplementedError()


@registry.metrics.register(
    "eds.ner_exact",
    deprecated=["eds.ner_exact_metric"],
)
class NerExactMetric(NerMetric):
    r"""
    The `eds.ner_exact` metric
    scores the extracted entities (that may be overlapping or nested)
    by looking in the spans returned by a given SpanGetter object and
    comparing predicted spans to gold spans for **exact** boundary and label matches.

    Let's view these elements as collections of (span → label) and count how
    many of the predicted spans match the gold spans exactly (and vice versa):

    +----------------------------------------------+--------------------------------------------+
    | pred                                         | ref                                        |
    +==============================================+============================================+
    | *La*{.chip .fp data-chip=PER}<br/>           | *patiente*{.chip .tp data-chip=PER}<br/>   |
    | *patiente*{.chip .tp data-chip=PER}<br/>     | *une fièvre*{.chip .fp data-chip=DIS}<br/> |
    | *fièvre aiguë*{.chip .fp data-chip=DIS}<br/> |                                            |
    +----------------------------------------------+--------------------------------------------+

    Precision, Recall and F1 (micro-average and per‐label) are computed as follows:

    - Precision: `p = |matched items of pred| / |pred|`
    - Recall: `r = |matched items of ref| / |ref|`
    - F1: `f = 2 / (1/p + 1/f)`

    Examples
    --------

    ```python
    from edsnlp.metrics.ner import NerExactMetric

    metric = NerExactMetric(span_getter=conv.span_setter, micro_key="micro")
    metric([ref], [pred])
    # Out: {
    #   'micro': {'f': 0.4, 'p': 0.33, 'r': 0.5, 'tp': 1, 'support': 2, 'positives': 3},
    #   'PER': {'f': 0.67, 'p': 0.5, 'r': 1, 'tp': 1, 'support': 1, 'positives': 2},
    #   'DIS': {'f': 0.0, 'p': 0.0, 'r': 0.0, 'tp': 0, 'support': 1, 'positives': 1},
    # }
    ```

    Parameters
    ----------
    span_getter: SpanGetter
        The span getter to use to extract the spans from the document
    micro_key: str
        The key to use to store the micro-averaged results for spans of all types
    filter_expr: str
        The filter expression to use to filter the documents. Evaluated with `doc` as the variable.
    """  # noqa: E501

    def __init__(
        self,
        span_getter: SpanGetterArg,
        micro_key: str = "micro",
        filter_expr: Optional[str] = None,
    ):
        self.span_getter = span_getter
        self.micro_key = micro_key
        self.filter_expr = filter_expr

    __init__.__doc__ = ner_exact_metric.__doc__

    def __call__(self, *examples):
        return ner_exact_metric(
            examples,
            span_getter=self.span_getter,
            micro_key=self.micro_key,
            filter_expr=self.filter_expr,
        )


@registry.metrics.register(
    "eds.ner_token",
    deprecated=["eds.ner_token_metric"],
)
class NerTokenMetric(NerMetric):
    r"""
    The `eds.ner_token` metric
    scores the extracted entities that may be overlapping or nested by looking in
    `doc.ents`, and `doc.spans`, and comparing the predicted and gold entities at the
    **token** level.

    Assuming we use the `eds` (or `fr` or `en`) tokenizer, in the above example, there
    are 3 annotated tokens in the reference, and 4 annotated tokens in the prediction.
    Let's view these elements as sets of (token, label) and count how many of the
    predicted tokens match the gold tokens exactly (and vice versa):

    +------------------------------------------+------------------------------------------+
    | pred                                     | ref                                      |
    +==========================================+==========================================+
    | *La*{.chip .fp data-chip=PER}<br/>       | *patiente*{.chip .tp data-chip=PER}<br/> |
    | *patiente*{.chip .tp data-chip=PER}<br/> | *une*{.chip .fp data-chip=DIS}<br/>      |
    | *fièvre*{.chip .tp data-chip=DIS}<br/>   | *fièvre*{.chip .tp data-chip=DIS}        |
    | *aiguë*{.chip .fp data-chip=DIS}         |                                          |
    +------------------------------------------+------------------------------------------+

    Precision, Recall and F1 (micro-average and per‐label) are computed as follows:

    - Precision: `p = |matched items of pred| / |pred|`
    - Recall: `r = |matched items of ref| / |ref|`
    - F1: `f = 2 / (1/p + 1/f)`

    Examples
    --------

    ```python
    from edsnlp.metrics.ner import NerTokenMetric

    metric = NerTokenMetric(span_getter=conv.span_setter, micro_key="micro")
    metric([ref], [pred])
    # Out: {
    #   'micro': {'f': 0.57, 'p': 0.5, 'r': 0.67, 'tp': 2, 'support': 3, 'positives': 4},
    #   'PER': {'f': 0.67, 'p': 0.5, 'r': 1, 'tp': 1, 'support': 1, 'positives': 2},
    #   'DIS': {'f': 0.5, 'p': 0.5, 'r': 0.5, 'tp': 1, 'support': 2, 'positives': 2}
    # }
    ```

    Parameters
    ----------
    span_getter: SpanGetter
        The span getter to use to extract the spans from the document
    micro_key: str
        The key to use to store the micro-averaged results for spans of all types
    filter_expr: str
        The filter expression to use to filter the documents. Will be evaluated
        with `doc` as the variable name, so you can use `doc.ents`, `doc.spans`, etc.
    """  # noqa: E501

    def __init__(
        self,
        span_getter: SpanGetterArg,
        micro_key: str = "micro",
        filter_expr: Optional[str] = None,
    ):
        self.span_getter = span_getter
        self.micro_key = micro_key
        self.filter_expr = filter_expr

    __init__.__doc__ = ner_token_metric.__doc__

    def __call__(self, *examples):
        return ner_token_metric(
            examples,
            span_getter=self.span_getter,
            micro_key=self.micro_key,
            filter_expr=self.filter_expr,
        )


@registry.metrics.register(
    "eds.ner_overlap",
    deprecated=["eds.ner_overlap_metric"],
)
class NerOverlapMetric(NerMetric):
    r"""
    The `eds.ner_overlap` metric
    scores the extracted entities that may be overlapping or nested
    by looking in the spans returned by a given SpanGetter object and
    counting a prediction as correct if it overlaps by at least the given
    Dice‐coefficient threshold with a gold span of the same label.

    This metric is useful for evaluating NER systems where the exact boundaries
    do not matter too much, but the presence of the entity at the same spot is important.
    For instance, you may not want to penalize a system that forgets determiners if
    the rest of the entity is correctly identified.

    Let's view these elements as sets of (span → label) and count how many of the
    predicted spans match the gold spans by at least the given Dice coefficient
    (and vice versa):

    +---------------------------------------------+------------------------------------------+
    | pred                                        | ref                                      |
    +=============================================+==========================================+
    | *La*{.chip .fp data-chip=PER}<br/>          | *patiente*{.chip .tp data-chip=PER}<br/> |
    | *patiente*{.chip .tp data-chip=PER}<br/>    | *une fièvre*{.chip .tp data-chip=DIS}    |
    | *fièvre aiguë*{.chip .tp data-chip=DIS}<br/>|                                          |
    +---------------------------------------------+------------------------------------------+

    Precision, Recall and F1 (micro-average and per‐label) are computed as follows:

    - Precision: `p = |matched items of pred| / |pred|`
    - Recall: `r = |matched items of ref| / |ref|`
    - F1: `f = 2 / (1/p + 1/f)`

    !!! note "Overlap threshold"

        The threshold is the minimum Dice coefficient to consider two spans as overlapping. Setting
        it to 1.0 will yield the same results as the `eds.ner_exact` metric, while setting it to a
        near-zero value (e.g., like 1e-14) will match any two spans that share at least one token.

    Examples
    --------

    ```python
    from edsnlp.metrics.ner import NerOverlapMetric

    metric = NerOverlapMetric(
        span_getter=conv.span_setter, micro_key="micro", threshold=0.5
    )
    metric([ref], [pred])
    # Out: {
    #   'micro': {'f': 0.8, 'p': 0.67, 'r': 1.0, 'tp': 2, 'support': 2, 'positives': 3},
    #   'PER': {'f': 0.67, 'p': 0.5, 'r': 1.0, 'tp': 1, 'support': 1, 'positives': 2},
    #   'DIS': {'f': 1.0, 'p': 1.0, 'r': 1.0, 'tp': 1, 'support': 1, 'positives': 1}
    # }
    ```

    Parameters
    ----------
    span_getter: SpanGetter
        The span getter to use to extract the spans from the document
    micro_key: str
        The key to use to store the micro-averaged results for spans of all types
    filter_expr: str
        The filter expression to use to filter the documents
    threshold: float
        The threshold on the Dice coefficient to consider two spans as overlapping
    """  # noqa: E501

    def __init__(
        self,
        span_getter: SpanGetterArg,
        micro_key: str = "micro",
        filter_expr: Optional[str] = None,
        threshold: float = 0.5,
    ):
        self.span_getter = span_getter
        self.micro_key = micro_key
        self.filter_expr = filter_expr
        self.threshold = threshold

    __init__.__doc__ = ner_overlap_metric.__doc__

    def __call__(self, *examples):
        return ner_overlap_metric(
            examples,
            span_getter=self.span_getter,
            micro_key=self.micro_key,
            filter_expr=self.filter_expr,
            threshold=self.threshold,
        )


# For backward compatibility
ner_exact_scorer = ner_exact_metric
ner_token_scorer = ner_token_metric
ner_overlap_scorer = ner_overlap_metric
create_ner_exact_scorer = NerExactScorer = NerExactMetric
create_ner_token_scorer = NerTokenScorer = NerTokenMetric
create_ner_overlap_scorer = NerOverlapScorer = NerOverlapMetric

__all__ = [
    "NerMetric",
    "NerExactMetric",
    "NerTokenMetric",
    "NerOverlapMetric",
    "ner_exact_metric",
    "ner_token_metric",
    "ner_overlap_metric",
]
