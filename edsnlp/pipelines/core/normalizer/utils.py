from typing import List, Tuple


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
        List of `(old, new)` tuples. `old` can list multiple characters.

    Returns
    -------
    str
        Processed text.
    """

    for olds, new in rep:
        for old in olds:
            text = text.replace(old, new)
    return text
