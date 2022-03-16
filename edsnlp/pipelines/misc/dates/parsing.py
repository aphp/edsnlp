import re
from typing import Callable, Dict, Optional

from edsnlp.pipelines.core.normalizer.accents.patterns import accents
from edsnlp.pipelines.core.normalizer.utils import replace

from .patterns.atomic import days, months


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


def time2int_factory(patterns: Dict[str, int]) -> Callable[[str], int]:
    """
    Factory for a `time2int` conversion function.

    Parameters
    ----------
    patterns : Dict[str, int]
        Dictionary of conversion/pattern.

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

        assert m is not None
        return m

    return time2int


month2int = time2int_factory(months.letter_months_dict)
day2int = time2int_factory(days.letter_days_dict)


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
        m = str2int(time)

        if m is not None:
            return m

        s = time.lower()
        s = replace(text=s, rep=accents)
        s = re.sub("[^a-z]", "", s)

        return patterns.get(s)

    return time2int


month2int_fast = time2int_fast_factory(months.letter_months_dict_simple)
day2int_fast = time2int_fast_factory(days.letter_days_dict_simple)
