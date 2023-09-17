from typing import Any, Callable, Dict, Iterable, Union

from spacy.tokens import Doc
from spacy.training import Example

Scorer = Union[
    Callable[[Iterable[Doc], Iterable[Doc]], Dict[str, Any]],
    Callable[[Iterable[Example]], Dict[str, Any]],
]
