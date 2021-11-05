from typing import List, Tuple

from spacy.tokens import Token


def replace(
    text: str,
    rep: List[Tuple[str, str]],
) -> str:
    """
    Replaces a list of characters in a given text.

    Parameters
    ----------
    text : str
        Text to modify.
    rep : List[Tuple[str, str]]
        List of ``(old, new)`` tuples. ``old`` can list multiple characters.

    Returns
    -------
    str
        Processed text.
    """

    for olds, new in rep:
        for old in olds:
            text = text.replace(old, new)
    return text


def first_normalization(token: Token) -> None:
    """
    Adds the first normalisation to the token. Should the custom attribute
    ``normalization`` be empty, it gets populated with ``token.text``
    (ie the verbatim text).

    Parameters
    ----------
    token : Token
        Token whose normalization is added.
    """
    if token._.normalization is None:
        token._.normalization = token.text
