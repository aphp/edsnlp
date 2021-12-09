from typing import List


def make_pattern(
    patterns: List[str],
    with_breaks: bool = False,
) -> str:
    """
    Create OR pattern from a list of patterns.

    Parameters
    ----------
    patterns : List[str]
        List of patterns to merge.
    with_breaks : bool, optional
        Whether to add breaks (``\b``) on each side, by default False

    Returns
    -------
    str
        Merged pattern.
    """

    pattern = "(" + "|".join(patterns) + ")"

    if with_breaks:
        pattern = r"\b" + pattern + r"\b"

    return pattern
