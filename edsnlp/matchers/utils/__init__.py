from typing import Dict, List, Union

ListOrStr = Union[List[str], str]
DictOrPattern = Union[str, List[str], Dict[str, Union[str, List[str]]]]
Patterns = Dict[str, DictOrPattern]

ATTRIBUTES = {
    "LOWER": "lower_",
    "TEXT": "text",
    "NORM": "norm_",
    "SHAPE": "shape_",
}

from .offset import alignment, offset  # noqa: E402, F401
from .text import get_text  # noqa: E402, F401
