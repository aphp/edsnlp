from functools import lru_cache
from typing import Dict, List, Optional, Union

from spacy.tokens import Doc, Span

ListOrStr = Union[List[str], str]
DictOrPattern = Union[Dict[str, ListOrStr], ListOrStr]
Patterns = Dict[str, DictOrPattern]

ATTRIBUTES = {
    "LOWER": "lower_",
    "TEXT": "text",
    "NORM": "norm_",
}


def make_pattern(
    patterns: List[str],
    with_breaks: bool = False,
    name: Optional[str] = None,
) -> str:
    """
    Create OR pattern from a list of patterns.

    Parameters
    ----------
    patterns : List[str]
        List of patterns to merge.
    with_breaks : bool, optional
        Whether to add breaks (``\b``) on each side, by default False
    name: str, optional
        Name of the group, using regex ``?P<>`` directive.

    Returns
    -------
    str
        Merged pattern.
    """

    if name:
        prefix = f"(?P<{name}>"
    else:
        prefix = "("

    # Sorting by length might be more efficient
    patterns.sort(key=len, reverse=True)

    pattern = prefix + "|".join(patterns) + ")"

    if with_breaks:
        pattern = r"\b" + pattern + r"\b"

    return pattern


@lru_cache(32)
def get_text(
    doclike: Union[Doc, Span], attr: str, ignore_excluded: bool = False
) -> str:
    """
    Get text using a custom attribute, possibly ignoring excluded tokens.

    Parameters
    ----------
    doclike : Union[Doc, Span]
        Doc or Span to get text from.
    attr : str
        Attribute to use.
    ignore_excluded : bool, optional
        Whether to skip excluded tokens, by default False

    Returns
    -------
    str
        Extracted text.
    """

    attr = attr.upper()

    if not ignore_excluded:
        if attr == "TEXT":
            return doclike.text
        elif attr == "LOWER":
            return doclike.text.lower()
        else:
            tokens = doclike
    else:
        tokens = [t for t in doclike if not t._.excluded]

    attr = ATTRIBUTES.get(attr, attr)

    if attr.startswith("_"):
        attr = attr[1:].lower()
        return "".join([getattr(t._, attr) + t.whitespace_ for t in tokens])
    else:
        return "".join([getattr(t, attr) + t.whitespace_ for t in tokens])
