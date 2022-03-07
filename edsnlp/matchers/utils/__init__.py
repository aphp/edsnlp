from typing import Dict, List, Union

ListOrStr = Union[List[str], str]
DictOrPattern = Union[Dict[str, ListOrStr], ListOrStr]
Patterns = Dict[str, DictOrPattern]

ATTRIBUTES = {
    "LOWER": "lower_",
    "TEXT": "text",
    "NORM": "norm_",
    "SHAPE": "shape_",
}

from .offset import alignment, offset
from .text import get_text
