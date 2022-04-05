from itertools import chain
from typing import List

from edsnlp.utils.regex import make_pattern

from .atomic.days import (
    day_pattern,
    letter_day_pattern,
    lz_numeric_day_pattern,
    numeric_day_pattern,
)
from .atomic.months import (
    letter_month_pattern,
    lz_numeric_month_pattern,
    month_pattern,
    numeric_month_pattern,
)
from .atomic.time import time_pattern
from .atomic.years import full_year_pattern as fy_pattern
from .atomic.years import year_pattern
from .directions import preceding_direction_pattern
from .relative import relative_patterns

raw_delimiters = [r"\/", r"\-"]
delimiters = raw_delimiters + [r"\.", r"[^\S\r\n]+"]

raw_delimiter_pattern = make_pattern(raw_delimiters)
raw_delimiter_with_spaces_pattern = make_pattern(raw_delimiters + [r"[^\S\r\n]+"])
delimiter_pattern = make_pattern(delimiters)

ante_num_pattern = f"(?<!.(?:{raw_delimiter_pattern})|[0-9][.,])"
post_num_pattern = f"(?!{raw_delimiter_pattern})"

full_year_pattern = ante_num_pattern + fy_pattern + post_num_pattern

absolute_date_pattern: List[str] = [
    ante_num_pattern
    + day_pattern
    + d
    + month_pattern
    + d
    + year_pattern
    + time_pattern
    + post_num_pattern
    for d in delimiters
] + [
    ante_num_pattern
    + year_pattern
    + d
    + numeric_month_pattern
    + d
    + numeric_day_pattern
    + time_pattern
    + post_num_pattern
    for d in delimiters
]

full_date_pattern = [
    ante_num_pattern
    + fy_pattern
    + d
    + lz_numeric_month_pattern
    + d
    + lz_numeric_day_pattern
    + post_num_pattern
    for d in [r"-", r"\."]
]
absolute_date_pattern.extend(full_date_pattern)

no_year_pattern = [
    day + raw_delimiter_with_spaces_pattern + month + time_pattern + post_num_pattern
    for day in [ante_num_pattern + numeric_day_pattern, letter_day_pattern]
    for month in [numeric_month_pattern + post_num_pattern, letter_month_pattern]
]
absolute_date_pattern.extend(no_year_pattern)

no_day_pattern = [
    letter_month_pattern
    + raw_delimiter_with_spaces_pattern
    + year_pattern
    + post_num_pattern,
    ante_num_pattern
    + lz_numeric_month_pattern
    + raw_delimiter_with_spaces_pattern
    + year_pattern
    + post_num_pattern,
]
absolute_date_pattern.extend(no_day_pattern)

absolute_date_pattern = [
    r"(?<=" + preceding_direction_pattern + r".{,3})?" + p
    for p in absolute_date_pattern
]


relative_date_patterns = relative_patterns


false_positive_pattern = make_pattern(
    [
        r"(\d+" + delimiter_pattern + r"){3,}\d+(?!:\d\d)\b",
        r"\d\/\d",
    ]
)
