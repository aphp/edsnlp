from typing import Any, Callable, Dict, Iterable, Union, Collection, Tuple

from spacy.tokens import Doc
from spacy.training import Example

import numpy as np

Examples = Union[Tuple[Iterable[Doc], Iterable[Doc]], Iterable[Example]]

Metric = Union[
    Callable[[Iterable[Doc], Iterable[Doc]], Dict[str, Dict[str, Any]]],
    Callable[[Iterable[Example]], Dict[str, Dict[str, Any]]],
]


def average_precision(pred: Dict[Any, float], gold: Iterable[Any]):
    # Average precision computation (pred is {prediction -> probability})
    pred = sorted(pred, key=lambda k: pred[k], reverse=True)
    correct = [p in gold for p in pred]
    cum_correct = np.cumsum(correct)
    num_gold = len(gold)
    precisions = cum_correct / np.arange(1, len(correct) + 1)
    recalls = cum_correct / num_gold if num_gold > 0 else np.zeros(len(correct))
    ap = 0.0
    for i in range(1, len(precisions)):
        if recalls[i] > recalls[i - 1]:
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def prf(pred: Collection, gold: Collection):
    tp = len(set(pred) & set(gold))
    num_pred = len(pred)
    num_gold = len(gold)
    return {
        "f": 2 * tp / max(1, num_pred + num_gold),
        "p": 1 if tp == num_pred else (tp / num_pred),
        "r": 1 if tp == num_gold else (tp / num_gold),
        "tp": tp,
        "support": num_gold,  # num gold
        "positives": num_pred,  # num predicted
    }


def make_examples(*examples):
    while isinstance(examples, tuple) and len(examples) == 1:
        examples = examples[0]
    if isinstance(examples, tuple) and len(examples) == 2:
        examples = (
            [Example(reference=g, predicted=p) for g, p in zip(*examples)]
            if len(examples) == 2
            else examples[0]
        )
    if not isinstance(examples, list):
        raise ValueError("Expected either a list of examples "
                         "or a tuple of two lists of docs")
    return examples
