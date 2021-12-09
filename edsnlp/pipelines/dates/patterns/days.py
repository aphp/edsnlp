from typing import List

letter_days: List[str] = [
    r"premier",
    r"1\s*er",
    "deux",
    "trois",
    "quatre",
    "cinq",
    "six",
    "sept",
    "huit",
    "neuf",
    "dix",
    "onze",
    "douze",
    "treize",
    "quatorze",
    "quinze",
    "seize",
    r"dix\-?\s*sept",
    r"dix\-?\s*huit",
    r"dix\-?\s*neuf",
    "vingt",
    r"vingt\-?\s*et\-?\s*un",
    r"vingt\-?\s*deux",
    r"vingt\-?\s*trois",
    r"vingt\-?\s*quatre",
    r"vingt\-?\s*cinq",
    r"vingt\-?\s*six",
    r"vingt\-?\s*sept",
    r"vingt\-?\s*huit",
    r"vingt\-?\s*neuf",
    r"trente",
    r"trente\-?\s*et\-?\s*un",
]

letter_day_pattern = "(" + "|".join(letter_days) + ")"

numeric_day_pattern = r"(0?[1-9]|[12]\d|3[0-1])"
numeric_day_pattern_with_leading_zero = r"(0?[1-9]|[12]\d|3[0-1])"

day_pattern = f"({letter_day_pattern}|{numeric_day_pattern})"
