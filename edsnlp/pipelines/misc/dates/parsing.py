import re
from typing import Callable, Dict, Optional

from edsnlp.pipelines.core.normalizer.accents.patterns import accents
from edsnlp.pipelines.core.normalizer.utils import replace

from .patterns.atomic import days, months
from .patterns.relative import relative_patterns


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

# warning: we reuse the function for the parsing of days,
# so we cannot parse numbers greater than 31.
letter_number_dict_simple = dict(
    {k: v for k, v in days.letter_days_dict_simple.items() if k != "premier"},
    **{"un": 1, "une": 1},
)
number2int_fast = time2int_fast_factory(letter_number_dict_simple)


def parse_relative(label, **kwargs: Dict[str, str]):
    res = dict()

    res["relative_direction"] = relative_patterns[label]["direction"]

    if "unit" in relative_patterns[label]:
        res["unit"] = relative_patterns[label]["unit"]
    else:
        u = kwargs["unit"]
        res["unit"] = process_unit(u)

    if "value" in relative_patterns[label]:
        res["value"] = relative_patterns[label]["value"]
    else:
        raw_v = kwargs["value"]

        # try to cast or parse the entire string as an int.
        # this doesn't work if there are several words (e.g. this fails: "environ 1")
        v = number2int_fast(raw_v)

        if v is None:
            res["unprocessed_value"] = raw_v
        else:
            res["value"] = v

    return res


dict_unit = {
    "an": "year",
    "annee": "year",
    "mois": "month",
    "semaine": "week",
    "jour": "day",
    "heure": "hour",
}


def process_unit(s: str):
    raw_s = s

    s = s.lower()
    s = replace(text=s, rep=accents)

    # remove the plural mark
    if s[-1] == "s" and s != ["mois"]:
        s = s[:-1]

    return dict_unit.get(s, raw_s)
