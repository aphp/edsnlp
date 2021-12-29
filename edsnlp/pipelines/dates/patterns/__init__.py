from itertools import chain
from typing import List

from edsnlp.matchers.utils import make_pattern

from .atomic.days import (
    day_pattern,
    letter_day_pattern,
    numeric_day_pattern,
    numeric_day_pattern_with_leading_zero,
)
from .atomic.months import (
    letter_month_pattern,
    month_pattern,
    numeric_month_pattern,
    numeric_month_pattern_with_leading_zero,
)
from .atomic.time import time_pattern
from .atomic.years import full_year_pattern as fy_pattern
from .atomic.years import year_pattern
from .current import current_pattern
from .relative import relative_pattern

raw_delimiters = [r"\/", r"\-"]
delimiters = raw_delimiters + [r"\.", r"[^\S\r\n]+"]

raw_delimiter_pattern = make_pattern(raw_delimiters)
raw_delimiter_with_spaces_pattern = make_pattern(raw_delimiters + [r"[^\S\r\n]+"])
delimiter_pattern = make_pattern(delimiters)

ante_num_pattern = f"(?<!{raw_delimiter_pattern})"
post_num_pattern = f"(?!{raw_delimiter_pattern})"

full_year_pattern = ante_num_pattern + fy_pattern + post_num_pattern

# Full dates
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
    + ante_num_pattern
    for d in delimiters
]

full_date_pattern = (
    ante_num_pattern
    + fy_pattern
    + "-"
    + numeric_month_pattern_with_leading_zero
    + "-"
    + numeric_day_pattern_with_leading_zero
    + post_num_pattern
)


no_year_pattern = [
    letter_day_pattern
    + raw_delimiter_with_spaces_pattern
    + letter_month_pattern
    + time_pattern,
    ante_num_pattern
    + numeric_day_pattern_with_leading_zero
    + raw_delimiter_with_spaces_pattern
    + numeric_month_pattern_with_leading_zero
    + time_pattern
    + post_num_pattern,
]

no_day_pattern = [
    letter_month_pattern
    + raw_delimiter_with_spaces_pattern
    + year_pattern
    + post_num_pattern,
    ante_num_pattern
    + numeric_month_pattern_with_leading_zero
    + raw_delimiter_with_spaces_pattern
    + year_pattern
    + post_num_pattern,
]

relative_date_pattern = relative_pattern

# TODO: add modifier patterns
since_pattern = [
    r"(?<=depuis)" + r".{,5}" + pattern
    for pattern in absolute_date_pattern
    + no_year_pattern
    + [
        full_date_pattern,
        relative_pattern,
    ]
]

false_positive_pattern = r"(\d+" + delimiter_pattern + r"){3,}\d+"
