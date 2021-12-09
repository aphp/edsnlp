from typing import List

from .patterns import (
    day_pattern,
    month_pattern,
    numeric_day_pattern,
    numeric_month_pattern,
    relative_pattern,
    year_pattern,
)
from .utils import make_pattern

delimiters = [r"\/", r"\.", r"\-", r"\s+"]

hour_pattern = r"([0-1]?\d|2[0-3])[h:][0-6]\d"

# Full dates
absolute_dates: List[str] = [
    day_pattern + d + month_pattern + d + year_pattern for d in delimiters
] + [
    year_pattern + d + numeric_month_pattern + d + numeric_day_pattern
    for d in delimiters
]
absolute_date_pattern = make_pattern(absolute_dates, with_breaks=True)
full_date_pattern = (
    year_pattern + "-" + numeric_month_pattern + "-" + numeric_day_pattern
)

no_year_dates = [day_pattern + d + month_pattern for d in delimiters]
no_year_dates.extend(
    [numeric_month_pattern + d + numeric_day_pattern for d in delimiters]
)
no_year_pattern = make_pattern(no_year_dates)

relative_date_pattern = relative_pattern

since_patterns = [
    r"(?<=depuis)" + r".{,5}" + pattern
    for pattern in [
        absolute_date_pattern,
        full_date_pattern,
        no_year_pattern,
        relative_pattern,
    ]
]
since_pattern = make_pattern(since_patterns)

false_positives = r"\b(?:\d\d[\s\.\/\-]){4,}|(?:\d\d{4,})"
