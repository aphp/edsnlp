from bisect import bisect_left
from functools import lru_cache, partial
from typing import List, Tuple

from spacy.tokens import Doc, Token

from . import ATTRIBUTES


def token_length(token: Token, custom: bool, attr: str):
    if custom:
        text = getattr(token._, attr)
    else:
        text = getattr(token, attr)
    return len(text)


@lru_cache(maxsize=32)
def alignment(
    doc: Doc,
    attr: str = "TEXT",
    ignore_excluded: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Align different representations of a `Doc` or `Span` object.

    Parameters
    ----------
    doc : Doc
        spaCy `Doc` or `Span` object
    attr : str, optional
        Attribute to use, by default `"TEXT"`
    ignore_excluded : bool, optional
        Whether to remove excluded tokens, by default True

    Returns
    -------
    Tuple[List[int], List[int]]
        An alignment tuple: original and clean lists.
    """
    assert isinstance(doc, Doc)

    attr = attr.upper()
    attr = ATTRIBUTES.get(attr, attr)

    custom = attr.startswith("_")

    if custom:
        attr = attr[1:].lower()

    # Define the length function
    length = partial(token_length, custom=custom, attr=attr)

    original = []
    clean = []

    cursor = 0

    for token in doc:

        if not ignore_excluded or token.tag_ != "EXCLUDED":

            # The token is not excluded, we add its extremities to the list
            original.append(token.idx)

            # We add the cursor
            clean.append(cursor)
            cursor += length(token)

            if token.whitespace_:
                cursor += 1

    return original, clean


def offset(
    doc: Doc,
    attr: str,
    ignore_excluded: bool,
    index: int,
) -> int:
    """
    Compute offset between the original text and a given representation
    (defined by the couple `attr`, `ignore_excluded`).

    The alignment itself is computed with
    [`alignment`][edsnlp.matchers.utils.offset.alignment].

    Parameters
    ----------
    doc : Doc
        The spaCy `Doc` object
    attr : str
        The attribute used by the [`RegexMatcher`][edsnlp.matchers.regex.RegexMatcher]
        (eg `NORM`)
    ignore_excluded : bool
        Whether the RegexMatcher ignores excluded tokens.
    index : int
        The index in the pre-processed text.

    Returns
    -------
    int
        The offset. To get the character index in the original document,
        just do: `#!python original = index + offset(doc, attr, ignore_excluded, index)`
    """
    original, clean = alignment(
        doc=doc,
        attr=attr,
        ignore_excluded=ignore_excluded,
    )

    # We use bisect to efficiently find the correct rightmost-lower index
    i = bisect_left(clean, index)
    i = min(i, len(original) - 1)

    return original[i] - clean[i]
