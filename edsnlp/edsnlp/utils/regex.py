import re
from typing import List, Optional

import regex


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


def compile_regex(reg: str, flags: re.RegexFlag):
    """
    This function tries to compile `reg`  using the `re` module, and
    fallbacks to the `regex` module that is more permissive.

    Parameters
    ----------
    reg: str

    Returns
    -------
    Union[re.Pattern, regex.Pattern]
    """
    try:
        return re.compile(reg, flags=flags)
    except re.error:
        try:
            return regex.compile(reg, flags=flags)
        except regex.error:
            raise Exception("Could not compile: {}".format(repr(reg)))
