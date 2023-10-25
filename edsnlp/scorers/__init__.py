from typing import Any, Callable, Dict, Iterable, Union

from spacy.tokens import Doc
from spacy.training import Example

Scorer = Union[
    Callable[[Iterable[Doc], Iterable[Doc]], Dict[str, Dict[str, Any]]],
    Callable[[Iterable[Example]], Dict[str, Dict[str, Any]]],
]


def prf(pred, gold):
    tp = len(set(pred) & set(gold))
    np = len(pred)
    ng = len(gold)
    return {
        "f": 2 * tp / max(1, np + ng),
        "p": 1 if tp == np else (tp / np),
        "r": 1 if tp == ng else (tp / ng),
        "tp": tp,
        "support": ng,  # num gold
        "positives": np,  # num predicted
    }


def make_examples(*args):
    if len(args) == 2:
        return (
            [Example(reference=g, predicted=p) for g, p in zip(*args)]
            if len(args) == 2
            else args[0]
        )
    else:
        raise ValueError("Expected either a list of examples or two lists of spans")
