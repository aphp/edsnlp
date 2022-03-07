from typing import Dict, List

from edsnlp.utils.regex import make_pattern

letter_days_dict: Dict[str, int] = {
    r"(premier|1\s*er)": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "onze": 11,
    "douze": 12,
    "treize": 13,
    "quatorze": 14,
    "quinze": 15,
    "seize": 16,
    r"dix\-?\s*sept": 17,
    r"dix\-?\s*huit": 18,
    r"dix\-?\s*neuf": 19,
    "vingt": 20,
    r"vingt\-?\s*et\-?\s*un": 21,
    r"vingt\-?\s*deux": 22,
    r"vingt\-?\s*trois": 23,
    r"vingt\-?\s*quatre": 24,
    r"vingt\-?\s*cinq": 25,
    r"vingt\-?\s*six": 26,
    r"vingt\-?\s*sept": 27,
    r"vingt\-?\s*huit": 28,
    r"vingt\-?\s*neuf": 29,
    r"trente": 30,
    r"trente\-?\s*et\-?\s*un": 31,
}

letter_days: List[str] = list(letter_days_dict.keys())

letter_day_pattern = make_pattern(letter_days)

numeric_day_pattern = r"(?<!\d)(0?[1-9]|[12]\d|3[01])(?!\d)"
lz_numeric_day_pattern = r"(?<!\d)(0[1-9]|[12]\d|3[01])(?!\d)"
nlz_numeric_day_pattern = r"(?<!\d)([1-9]|[12]\d|3[01])(?!\d)"

day_pattern = f"(?P<day>{letter_day_pattern}|{numeric_day_pattern})"

letter_day_pattern = f"(?P<day>{letter_day_pattern})"
numeric_day_pattern = f"(?P<day>{numeric_day_pattern})"
lz_numeric_day_pattern = f"(?P<day>{lz_numeric_day_pattern})"
