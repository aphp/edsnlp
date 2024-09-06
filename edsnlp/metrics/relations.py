from collections import defaultdict
from itertools import product
from typing import Any, Optional

from edsnlp import registry
from edsnlp.metrics import Examples, make_examples, prf
from edsnlp.utils.span_getters import RelationCandidateGetter, get_spans
from edsnlp.utils.typing import AsList


def relations_scorer(
    examples: Examples,
    candidate_getter: AsList[RelationCandidateGetter],
    micro_key: str = "micro",
    filter_expr: Optional[str] = None,
):
    """
    Compute relation precision and recall on gold entity spans

    Parameters
    ----------
    examples : Examples
        The examples to score, either a tuple of (golds, preds) or a list of
        spacy.training.Example objects
    candidate_getter : AsList[RelationCandidateGetter]
        The candidate getters to use to extract the possible relations from the
        documents. Each candidate getter should be a dictionary with the keys
        "head", "tail", and "labels". The "head" and "tail" keys should be
        SpanGetterArg objects, and the "labels" key should be a list of strings
        for these head-tail pairs. The optional "label_filter" maps head labels
        to allowed tail labels, unlisted head labels are unrestricted.
    micro_key : str
        The key to use to store the micro-averaged results for spans of all types
    filter_expr : Optional[str]
        The filter expression to use to filter the documents

    Returns
    -------
    Dict[str, Dict[str, float]]
        Precision, recall and F-score for each relation label and the micro average
    """
    examples = make_examples(examples)
    if filter_expr is not None:
        filter_fn = eval(f"lambda doc: {filter_expr}")
        examples = [eg for eg in examples if filter_fn(eg.reference)]
    # Predicted and gold relation tuples for each label
    annotations = defaultdict(lambda: (set(), set()))
    annotations[micro_key] = (set(), set())
    total_pred_count = 0
    total_gold_count = 0

    for candidate in candidate_getter:
        head_getter = candidate["head"]
        tail_getter = candidate["tail"]
        labels = candidate["labels"]
        symmetric = candidate.get("symmetric") or False
        label_filter = candidate.get("label_filter")
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
                if (
                    label_filter
                    and head.label_ in label_filter
                    and tail.label_ not in label_filter[head.label_]
                ):
                    continue
                total_pred_count += 1
                for label in labels:
                    if (
                        tail in head._.rel.get(label, ())
                        or symmetric
                        and head in tail._.rel.get(label, ())
                    ):
                        if symmetric and h_key > t_key:
                            h_key, t_key = t_key, h_key
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
                if (
                    label_filter
                    and head.label_ in label_filter
                    and tail.label_ not in label_filter[head.label_]
                ):
                    continue
                total_gold_count += 1
                for label in labels:
                    if (
                        tail in head._.rel.get(label, ())
                        or symmetric
                        and head in tail._.rel.get(label, ())
                    ):
                        if symmetric and h_key > t_key:
                            h_key, t_key = t_key, h_key
                        annotations[label][1].add((eg_idx, h_key, t_key, label))
                        annotations[micro_key][1].add((eg_idx, h_key, t_key, label))

    if total_pred_count != total_gold_count:
        raise ValueError(
            f"Number of predicted and gold candidate pairs differ: {total_pred_count} "
            f"!= {total_gold_count}. Make sure that you are running your "
            "relation detection pipe on the gold annotations, and not spans "
            "predicted by another NER pipe in your model."
        )

    return {name: prf(pred, gold) for name, (pred, gold) in annotations.items()}


@registry.metrics.register("eds.relations")
class RelationsMetric:
    def __init__(
        self,
        candidate_getter: AsList[RelationCandidateGetter],
        micro_key: str = "micro",
        filter_expr: Optional[str] = None,
    ):
        self.candidate_getter = candidate_getter
        self.micro_key = micro_key
        self.filter_expr = filter_expr

    __init__.__doc__ = relations_scorer.__doc__

    def __call__(self, *examples: Any):
        return relations_scorer(
            examples,
            candidate_getter=self.candidate_getter,
            micro_key=self.micro_key,
            filter_expr=self.filter_expr,
        )
