import re
from typing import Callable, Dict

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
        m = str2int(time)

        if m is not None:
            return m

        for pattern, key in patterns.items():
            if re.match(f"^{pattern}$", time):
                m = key
                break

        return m

    return time2int


month2int = time2int_factory(months.letter_months_dict)
day2int = time2int_factory(days.letter_days_dict)
