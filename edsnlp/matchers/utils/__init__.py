from typing import Dict, List, Union

ListOrStr = Union[List[str], str]
DictOrPattern = Union[Dict[str, ListOrStr], ListOrStr]
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

from .offset import alignment, offset
from .text import get_text
