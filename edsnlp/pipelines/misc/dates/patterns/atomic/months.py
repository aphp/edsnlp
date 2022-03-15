from typing import Dict, List

from edsnlp.utils.regex import make_pattern

letter_months_dict: Dict[str, int] = {
    r"(janvier|janv\.?)": 1,
    r"(f[ée]vrier|f[ée]v\.?)": 2,
    r"(mars|mar\.?)": 3,
    r"(avril|avr\.?)": 4,
    r"mai": 5,
    r"juin": 6,
    r"(juillet|juill?\.?)": 7,
    r"ao[uû]t": 8,
    r"(septembre|sept?\.?)": 9,
    r"(octobre|oct\.?)": 10,
    r"(novembre|nov\.)": 11,
    r"(d[ée]cembre|d[ée]c\.?)": 12,
}

letter_months: List[str] = list(letter_months_dict.keys())

letter_month_pattern = make_pattern(letter_months, with_breaks=True)

numeric_month_pattern = r"(?<!\d)(0?[1-9]|1[0-2])(?!\d)"
lz_numeric_month_pattern = r"(?<!\d)(0[1-9]|1[0-2])(?!\d)"

month_pattern = f"(?P<month>{letter_month_pattern}|{numeric_month_pattern})"
letter_month_pattern = f"(?P<month>{letter_month_pattern})"
numeric_month_pattern = f"(?P<month>{numeric_month_pattern})"
lz_numeric_month_pattern = f"(?P<month>{lz_numeric_month_pattern})"


list_month_long = [
    "janvier",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
]
list_month_short = [
    "janv",
    "fev",
    "mar",
    "avr",
    None,
    None,
    "juill",
    None,
    "sept",
    "oct",
    "nov",
    "dec",
]
letter_months_dict_simple = dict(
    {s: i for i, s in enumerate(list_month_short, 1) if s is not None},
    **{s: i for i, s in enumerate(list_month_long, 1) if s is not None},
)
