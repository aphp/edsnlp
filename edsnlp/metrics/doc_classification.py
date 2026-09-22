from typing import Any, Dict, Optional, Sequence, Set, Tuple

from edsnlp import registry
from edsnlp.metrics import Examples, make_examples, prf
from edsnlp.utils.typing import AsList

Item = Tuple[int, Any]


def _items(values: Sequence[Any]) -> Set[Item]:
    """
    Flatten per-document values into a set of `(doc index, label)` items.

    A document whose value is `None` contributes nothing, so a head that is not
    annotated on a given document is simply ignored.
    """
    items = set()
    for idx, value in enumerate(values):
        if value is None:
            continue
        if isinstance(value, (list, set, tuple)):
            items.update((idx, label) for label in value)
        else:
            items.add((idx, value))
    return items


def doc_classification_metric(
    examples: Examples,
    label_attr: AsList[str],
    micro_key: str = "micro",
    macro_key: str = "macro",
    filter_expr: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Parameters
    ----------
    examples: Examples
        The examples to score, either a tuple of (golds, preds) or a list of
        spacy.training.Example objects
    label_attr: AsList[str]
        The `Doc._` attributes to score, one per classification head. A single
        attribute can be passed directly, without wrapping it in a list
    micro_key: str
        The key to use to store the micro-averaged results
    macro_key: str
        The key to use to store the macro-averaged results
    filter_expr: Optional[str]
        The filter expression to use to filter the documents

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping each attribute to its per-label, micro and macro scores
    """
    examples = make_examples(examples)
    if filter_expr is not None:
        filter_fn = eval(f"lambda doc: {filter_expr}")
        examples = [eg for eg in examples if filter_fn(eg.reference)]

    metrics = {}

    for attr in label_attr:
        pred = _items([getattr(eg.predicted._, attr, None) for eg in examples])
        gold = _items([getattr(eg.reference._, attr, None) for eg in examples])
        labels = {label for _, label in (pred | gold)}

        attr_metrics = {
            label: prf(
                {item for item in pred if item[1] == label},
                {item for item in gold if item[1] == label},
            )
            for label in labels
        }
        attr_metrics[micro_key] = prf(pred, gold)
        attr_metrics[macro_key] = {
            key: sum(attr_metrics[label][key] for label in labels) / len(labels)
            if labels
            else 0.0
            for key in ("f", "p", "r")
        }
        attr_metrics[macro_key].update(support=len(gold), classes=len(labels))
        metrics[attr] = attr_metrics

    return metrics


@registry.metrics.register(
    "eds.doc_classification",
    deprecated=["eds.doc_classif"],
)
class DocClassificationMetric:
    """
    The `eds.doc_classification` metric
    evaluates document-level classification by comparing the predicted and gold
    values of one or several `Doc._` attributes. For each attribute it computes
    Precision, Recall, F1, number of true positives (tp), number of gold
    instances (support) and number of predicted instances (positives), per label
    and micro-averaged under `micro_key`. The per-label scores are also
    macro-averaged under `macro_key`.

    The metric adapts to the type of each attribute. An attribute holding a
    single value (as predicted by `eds.single_label_head`) contributes one item
    per document, so its micro scores amount to the accuracy of the head. An
    attribute holding a list or a set of values (as predicted by
    `eds.multi_label_head`) contributes one item per predicted label, and is
    therefore scored as a multi-label task.

    A document whose gold value is `None` for a given attribute is ignored for
    that attribute, which lets partially annotated corpora be scored as-is.

    ```python
    import edsnlp
    from edsnlp.metrics.doc_classification import DocClassificationMetric
    from spacy.tokens import Doc

    for attr in ("dp", "das"):
        if not Doc.has_extension(attr):
            Doc.set_extension(attr, default=None)

    nlp = edsnlp.blank("eds")
    gold, pred = nlp("Compte rendu."), nlp("Compte rendu.")
    gold._.dp, gold._.das = "C34", ["E11", "I10"]
    pred._.dp, pred._.das = "C34", ["E11", "N18"]

    metric = DocClassificationMetric(label_attr=["dp", "das"])
    scores = metric([gold], [pred])
    print(scores["dp"]["micro"]["f"])
    # Out: 1.0
    print(scores["das"]["micro"]["f"])
    # Out: 0.5
    ```
    """

    def __init__(
        self,
        label_attr: AsList[str],
        micro_key: str = "micro",
        macro_key: str = "macro",
        filter_expr: Optional[str] = None,
    ):
        self.label_attr = label_attr
        self.micro_key = micro_key
        self.macro_key = macro_key
        self.filter_expr = filter_expr

    __init__.__doc__ = doc_classification_metric.__doc__

    def __call__(self, *examples):
        return doc_classification_metric(
            examples,
            label_attr=self.label_attr,
            micro_key=self.micro_key,
            macro_key=self.macro_key,
            filter_expr=self.filter_expr,
        )


create_doc_classification_scorer = DocClassificationScorer = DocClassificationMetric

__all__ = [
    "doc_classification_metric",
    "DocClassificationMetric",
]
