from typing import List, Optional


def make_pattern(
    patterns: List[str],
    with_breaks: bool = False,
    name: Optional[str] = None,
) -> str:
    r"""
    Create OR pattern from a list of patterns.

    Parameters
    ----------
    patterns : List[str]
        List of patterns to merge.
    with_breaks : bool, optional
        Whether to add breaks (`\b`) on each side, by default False
    name: str, optional
        Name of the group, using regex `?P<>` directive.

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
