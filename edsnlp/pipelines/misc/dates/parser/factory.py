import re
from typing import Callable, Dict, Optional

from edsnlp.pipelines.core.normalizer.accents import Accents


def str2int(time: str) -> int:
    """
    Converts a string to an integer. Returns `None` if the string cannot be converted.

    Parameters
    ----------
    time : str
        String representation

    Returns
    -------
    int
        Integer conversion.
    """
    try:
        return int(time)
    except ValueError:
        return None


def time2int_factory(
    patterns: Dict[str, int],
    use_as_default: bool = False,
) -> Callable[[str], int]:
    """
    Factory for a `time2int` conversion function.

    Parameters
    ----------
    patterns : Dict[str, int]
        Dictionary of conversion/pattern.

    use_as_default : bool, by default `False`
        Whether to use the value as default.

    Returns
    -------
    Callable[[str], int]
        String to integer function.
    """

    def time2int(time: str) -> int:
        """
        Converts a string representation to the proper integer,
        iterating over a dictionnary of pattern/conversion.

        Parameters
        ----------
        time : str
            String representation

        Returns
        -------
        int
            Integer conversion
        """
        m = None

        for pattern, key in patterns.items():
            if re.match(f"^{pattern}$", time):
                m = key
                break

        if use_as_default:
            m = m or time
        assert m is not None
        return m

    return time2int


def time2int_fast_factory(patterns: Dict[str, int]) -> Callable[[str], Optional[int]]:
    """
    Factory for a `time2int_fast` conversion function.

    Parameters
    ----------
    patterns : Dict[str, int]
        Dictionary of conversion.

    Returns
    -------
    Callable[[str], int]
        String to integer function.
    """

    TRANSLATION_TABLE = Accents(None).translation_table
    PATTERNS = {k.translate(TRANSLATION_TABLE): v for k, v in patterns.items()}

    def time2int(time: str) -> int:
        """
        Try to convert a string representation to the proper
        integer using 2 fast methods:
        - casting the string to an int
        - using a simple dictionary access.

        Parameters
        ----------
        time : str
            String representation

        Returns
        -------
        int
            Integer conversion or None if the fast conversion failed
        """

        s = time.lower()
        s = s.translate(TRANSLATION_TABLE)
        s = re.sub("[^a-z]", "", s)

        return PATTERNS.get(s)

    return time2int
