from typing import Any, Callable, Dict, Iterable, Union

from spacy.tokens import Doc
from spacy.training import Example

Scorer = Union[
    Callable[[Iterable[Doc], Iterable[Doc]], Dict[str, Any]],
    Callable[[Iterable[Example]], Dict[str, Any]],
]


def make_examples(*args):
    if len(args) == 2:
        return (
            [Example(reference=g, predicted=p) for g, p in zip(*args)]
            if len(args) == 2
            else args[0]
        )
    else:
        raise ValueError("Expected either a list of examples or two lists of spans")
