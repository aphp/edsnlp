from typing import Dict, List, Union

ListOrStr = Union[List[str], str]
DictOrPattern = Union[str, List[str], Dict[str, Union[str, List[str]]]]
Patterns = Dict[str, DictOrPattern]


def normalize_token_attr(attr):
    if attr.startswith("doc.") or attr.startswith("span."):
        return None
    attr = attr.replace("token.", "")
    lower = attr.replace("_", "").lower()
    return "text" if lower == "orth" else lower


ATTRIBUTES = {
    "LOWER": "lower_",
    "TEXT": "text",
    "NORM": "norm_",
    "SHAPE": "shape_",
}

from .offset import alignment  # noqa: E402, F401
from .text import get_text  # noqa: E402, F401
