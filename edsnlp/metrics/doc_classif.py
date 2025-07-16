from typing import Any, Dict, Iterable, Optional, Tuple, Union

from spacy.tokens import Doc
from spacy.training import Example

from edsnlp import registry
from edsnlp.metrics import make_examples


def doc_classification_metric(
    examples: Union[Tuple[Iterable[Doc], Iterable[Doc]], Iterable[Example]],
    label_attr: str = "label",
    micro_key: str = "micro",
    filter_expr: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Scores document-level classification (accuracy, precision, recall, F1).

    Parameters
    ----------
    examples: Examples
        The examples to score, either a tuple of (golds, preds) or a list of
        spacy.training.Example objects
    label_attr: str
        The Doc._ attribute containing the label
    micro_key: str
        The key to use to store the micro-averaged results
    filter_expr: str
        The filter expression to use to filter the documents

    Returns
    -------
    Dict[str, Any]
    """
    examples = make_examples(examples)
    if filter_expr is not None:
        filter_fn = eval(f"lambda doc: {filter_expr}")
        examples = [eg for eg in examples if filter_fn(eg.reference)]

    pred_labels = []
    gold_labels = []
    for eg in examples:
        pred = getattr(eg.predicted._, label_attr, None)
        gold = getattr(eg.reference._, label_attr, None)
        pred_labels.append(pred)
        gold_labels.append(gold)

    labels = set(gold_labels) | set(pred_labels)
    results = {}
    for label in labels:
        pred_set = [i for i, p in enumerate(pred_labels) if p == label]
        gold_set = [i for i, g in enumerate(gold_labels) if g == label]
        tp = len(set(pred_set) & set(gold_set))
        num_pred = len(pred_set)
        num_gold = len(gold_set)
        results[label] = {
            "f": 2 * tp / max(1, num_pred + num_gold),
            "p": 1 if tp == num_pred else (tp / num_pred) if num_pred else 0.0,
            "r": 1 if tp == num_gold else (tp / num_gold) if num_gold else 0.0,
            "tp": tp,
            "support": num_gold,
            "positives": num_pred,
        }

    tp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == g)
    num_pred = len(pred_labels)
    num_gold = len(gold_labels)
    results[micro_key] = {
        "accuracy": tp / num_gold if num_gold else 0.0,
        "f": 2 * tp / max(1, num_pred + num_gold),
        "p": tp / num_pred if num_pred else 0.0,
        "r": tp / num_gold if num_gold else 0.0,
        "tp": tp,
        "support": num_gold,
        "positives": num_pred,
    }
    return results


@registry.metrics.register("eds.doc_classification")
class DocClassificationMetric:
    def __init__(
        self,
        label_attr: str = "label",
        micro_key: str = "micro",
        filter_expr: Optional[str] = None,
    ):
        self.label_attr = label_attr
        self.micro_key = micro_key
        self.filter_expr = filter_expr

    def __call__(self, *examples):
        return doc_classification_metric(
            examples,
            label_attr=self.label_attr,
            micro_key=self.micro_key,
            filter_expr=self.filter_expr,
        )


__all__ = [
    "doc_classification_metric",
    "DocClassificationMetric",
]
