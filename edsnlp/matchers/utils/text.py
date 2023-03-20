from functools import lru_cache
from typing import Union

from spacy.tokens import Doc, Span

from . import ATTRIBUTES


@lru_cache(32)
def get_text(
    doclike: Union[Doc, Span],
    attr: str,
    ignore_excluded: bool,
    ignore_space_tokens: bool = False,
) -> str:
    """
    Get text using a custom attribute, possibly ignoring excluded tokens.

    Parameters
    ----------
    doclike : Union[Doc, Span]
        Doc or Span to get text from.
    attr : str
        Attribute to use.
    ignore_excluded : bool
        Whether to skip excluded tokens, by default False
    ignore_space_tokens : bool
        Whether to skip space tokens, by default False

    Returns
    -------
    str
        Extracted text.
    """

    attr = attr.upper()

    if not ignore_excluded and not ignore_space_tokens:
        if attr == "TEXT":
            return doclike.text
        elif attr == "LOWER":
            return doclike.text.lower()
        else:
            tokens = doclike
    else:
        if not ignore_space_tokens:
            tokens = [t for t in doclike if t.tag_ != "EXCLUDED"]
        else:
            tokens = [t for t in doclike if t.tag_ not in ("EXCLUDED", "SPACE")]

    if not tokens:
        return ""

    attr = ATTRIBUTES.get(attr, attr)

    if attr.startswith("_"):
        attr = attr[1:].lower()
        return "".join(
            [getattr(t._, attr) + t.whitespace_ for t in tokens[:-1]]
        ) + getattr(tokens[-1], attr)
    else:
        return "".join(
            [getattr(t, attr) + t.whitespace_ for t in tokens[:-1]]
        ) + getattr(tokens[-1], attr)
