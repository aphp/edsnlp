from typing import Any, Optional

from edsnlp import registry
from edsnlp.metrics import Examples, make_examples, prf


def dependency_parsing_metric(
    examples: Examples,
    filter_expr: Optional[str] = None,
):
    """
    Compute the UAS and LAS scores for dependency parsing.

    Parameters
    ----------
    examples : Examples
        The examples to score, either a tuple of (golds, preds) or a list of
        spacy.training.Example objects
    filter_expr : Optional[str]
        The filter expression to use to filter the documents

    Returns
    -------
    Dict[str, float]
    """
    items = {
        "uas": (set(), set()),
        "las": (set(), set()),
    }
    examples = make_examples(examples)
    if filter_expr is not None:
        filter_fn = eval(f"lambda doc: {filter_expr}")
        examples = [eg for eg in examples if filter_fn(eg.reference)]

    for eg_idx, eg in enumerate(examples):
        for token in eg.reference:
            items["uas"][0].add((eg_idx, token.i, token.head.i))
            items["las"][0].add((eg_idx, token.i, token.head.i, token.dep_))

        for token in eg.predicted:
            items["uas"][1].add((eg_idx, token.i, token.head.i))
            items["las"][1].add((eg_idx, token.i, token.head.i, token.dep_))

    return {name: prf(pred, gold)["f"] for name, (pred, gold) in items.items()}


@registry.metrics.register("eds.dep_parsing")
class DependencyParsingMetric:
    def __init__(self, filter_expr: Optional[str] = None):
        self.filter_expr = filter_expr

    __init__.__doc__ = dependency_parsing_metric.__doc__

    def __call__(self, *examples: Any):
        return dependency_parsing_metric(examples, self.filter_expr)
