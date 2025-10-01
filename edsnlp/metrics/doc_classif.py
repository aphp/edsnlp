from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from spacy.tokens import Doc
from spacy.training import Example

from edsnlp import registry
from edsnlp.metrics import make_examples


def doc_classification_metric(
    examples: Union[Tuple[Iterable[Doc], Iterable[Doc]], Iterable[Example]],
    label_attr: List[str],
    micro_key: str = "micro",
    macro_key: str = "macro",
    filter_expr: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Scores multi-head document-level classification (accuracy, precision, recall, F1)
    for each head.

    Parameters
    ----------
    examples: Examples
        The examples to score, either a tuple of (golds, preds) or a list of
        spacy.training.Example objects
    label_attr: List[str]
        The list of Doc._ attributes containing the labels for each head
    micro_key: str
        The key to use to store the micro-averaged results
    macro_key: str
        The key to use to store the macro-averaged results
    filter_expr: str
        The filter expression to use to filter the documents

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping head names to their respective metrics
    """
    examples = make_examples(examples)
    if filter_expr is not None:
        filter_fn = eval(f"lambda doc: {filter_expr}")
        examples = [eg for eg in examples if filter_fn(eg.reference)]

    all_head_results = {}

    for head_name in label_attr:
        pred_labels = []
        gold_labels = []

        for eg in examples:
            pred = getattr(eg.predicted._, head_name, None)
            gold = getattr(eg.reference._, head_name, None)
            pred_labels.append(pred)
            gold_labels.append(gold)

        labels = set(gold_labels) | set(pred_labels)
        labels = {label for label in labels if label is not None}
        head_results = {}

        for label in labels:
            tp = sum(
                1 for p, g in zip(pred_labels, gold_labels) if p == label and g == label
            )
            fp = sum(
                1 for p, g in zip(pred_labels, gold_labels) if p == label and g != label
            )
            fn = sum(
                1 for p, g in zip(pred_labels, gold_labels) if g == label and p != label
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            head_results[label] = {
                "f": f1,
                "p": precision,
                "r": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": tp + fn,
                "positives": tp + fp,
            }

        total_tp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == g)
        total_fp = sum(1 for p, g in zip(pred_labels, gold_labels) if p != g)
        total_fn = total_fp

        micro_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        micro_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        )
        micro_f1 = (
            (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )
        accuracy = total_tp / len(pred_labels) if len(pred_labels) > 0 else 0.0

        head_results[micro_key] = {
            "accuracy": accuracy,
            "f": micro_f1,
            "p": micro_precision,
            "r": micro_recall,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "support": len(gold_labels),
            "positives": len(pred_labels),
        }

        per_class_precisions = [head_results[label]["p"] for label in labels]
        per_class_recalls = [head_results[label]["r"] for label in labels]
        per_class_f1s = [head_results[label]["f"] for label in labels]

        macro_precision = (
            sum(per_class_precisions) / len(per_class_precisions)
            if per_class_precisions
            else 0.0
        )
        macro_recall = (
            sum(per_class_recalls) / len(per_class_recalls)
            if per_class_recalls
            else 0.0
        )
        macro_f1 = sum(per_class_f1s) / len(per_class_f1s) if per_class_f1s else 0.0

        head_results[macro_key] = {
            "f": macro_f1,
            "p": macro_precision,
            "r": macro_recall,
            "support": len(labels),
            "classes": len(labels),
        }

        all_head_results[head_name] = head_results

    return all_head_results


@registry.metrics.register("eds.doc_classification")
class DocClassificationMetric:
    def __init__(
        self,
        label_attr: List[str],
        micro_key: str = "micro",
        macro_key: str = "macro",
        filter_expr: Optional[str] = None,
    ):
        """
        Multi-head document classification metric.

        Parameters
        ----------
        label_attr: List[str]
            List of Doc._ attributes containing the labels for each head
        micro_key: str
            The key to use to store the micro-averaged results
        macro_key: str
            The key to use to store the macro-averaged results
        filter_expr: str
            The filter expression to use to filter the documents
        """
        self.label_attr = label_attr
        self.micro_key = micro_key
        self.macro_key = macro_key
        self.filter_expr = filter_expr

    def __call__(self, *examples):
        return doc_classification_metric(
            examples,
            label_attr=self.label_attr,
            micro_key=self.micro_key,
            macro_key=self.macro_key,
            filter_expr=self.filter_expr,
        )


__all__ = [
    "doc_classification_metric",
    "DocClassificationMetric",
]
