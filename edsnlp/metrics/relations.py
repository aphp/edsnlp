from collections import defaultdict
from itertools import product
from typing import Any, Optional

from edsnlp import registry
from edsnlp.metrics import Examples, make_examples, prf
from edsnlp.utils.span_getters import SpanGetterArg, get_spans
from edsnlp.utils.typing import AsList


def relations_scorer(
    examples: Examples,
    head_getter: SpanGetterArg,
    tail_getter: SpanGetterArg,
    labels: AsList[str],
    micro_key: str = "micro",
    filter_expr: Optional[str] = None,
):
    """
    Scores the attributes predictions between a list of gold and predicted spans.

    Parameters
    ----------
    examples : Examples
        The examples to score, either a tuple of (golds, preds) or a list of
        spacy.training.Example objects
    head_getter : SpanGetterArg
        The span getter to use to extract the relation heads from the document
    tail_getter : SpanGetterArg
        The span getter to use to extract the relation tails from the document
    labels : Sequence[str]
        The labels of the relations to evaluate
    micro_key : str
        The key to use to store the micro-averaged results for spans of all types
    filter_expr : Optional[str]
        The filter expression to use to filter the documents

    Returns
    -------
    Dict[str, float]
    """
    examples = make_examples(examples)
    if filter_expr is not None:
        filter_fn = eval(f"lambda doc: {filter_expr}")
        examples = [eg for eg in examples if filter_fn(eg.reference)]
    # annotations: {label -> preds, golds, pred_with_probs}
    annotations = defaultdict(lambda: (set(), set(), dict()))
    annotations[micro_key] = (set(), set(), dict())
    total_pred_count = 0
    total_gold_count = 0

    for eg_idx, eg in enumerate(examples):
        pred_heads = [
            ((h.start, h.end, h.label_), h)
            for h in get_spans(eg.predicted, head_getter)
        ]
        pred_tails = [
            ((t.start, t.end, t.label_), t)
            for t in get_spans(eg.predicted, tail_getter)
        ]
        for (h_key, head), (t_key, tail) in product(pred_heads, pred_tails):
            total_pred_count += 1
            for label in labels:
                if tail in head._.rel.get(label, ()):
                    annotations[label][0].add((eg_idx, h_key, t_key, label))
                    annotations[micro_key][0].add((eg_idx, h_key, t_key, label))

        gold_heads = [
            ((h.start, h.end, h.label_), h)
            for h in get_spans(eg.reference, head_getter)
        ]
        gold_tails = [
            ((t.start, t.end, t.label_), t)
            for t in get_spans(eg.reference, tail_getter)
        ]
        for (h_key, head), (t_key, tail) in product(gold_heads, gold_tails):
            total_gold_count += 1
            for label in labels:
                if tail in head._.rel.get(label, ()):
                    annotations[label][1].add((eg_idx, h_key, t_key, label))
                    annotations[micro_key][1].add((eg_idx, h_key, t_key, label))

    if total_pred_count != total_gold_count:
        raise ValueError(
            f"Number of predicted and gold candidate pairs differ: {total_pred_count} "
            f"!= {total_gold_count}. Make sure that you are running your span "
            "attribute classification pipe on the gold annotations, and not spans "
            "predicted by another NER pipe in your model."
        )

    return {
        name: {
            **prf(pred, gold),
            # "ap": average_precision(pred_with_prob, gold),
        }
        for name, (pred, gold, pred_with_prob) in annotations.items()
    }


@registry.metrics.register("eds.relations")
class RelationsMetric:
    def __init__(
        self,
        head_getter: SpanGetterArg,
        tail_getter: SpanGetterArg,
        labels: AsList[str],
        micro_key: str = "micro",
        filter_expr: Optional[str] = None,
    ):
        self.head_getter = head_getter
        self.tail_getter = tail_getter
        self.labels = labels
        self.micro_key = micro_key
        self.filter_expr = filter_expr

    __init__.__doc__ = relations_scorer.__doc__

    def __call__(self, *examples: Any):
        return relations_scorer(
            examples,
            head_getter=self.head_getter,
            tail_getter=self.tail_getter,
            labels=self.labels,
            micro_key=self.micro_key,
            filter_expr=self.filter_expr,
        )