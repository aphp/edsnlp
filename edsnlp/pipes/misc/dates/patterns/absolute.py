from .atomic.days import (  # lz_numeric_day_pattern,
    day_pattern,
    letter_day_pattern,
    numeric_day_pattern,
)
from .atomic.delimiters import (
    ante_num_pattern,
    delimiters,
    post_num_pattern,
    post_num_with_letter_pattern,
    raw_delimiter_with_spaces_pattern,
    raw_delimiters,
)
from .atomic.modes import mode_pattern
from .atomic.months import (
    letter_month_pattern,
    lz_numeric_month_pattern,
    month_pattern,
    numeric_month_pattern,
)
from .atomic.time import time_pattern
from .atomic.years import full_year_pattern as fy_pattern
from .atomic.years import year_pattern

absolute_pattern = []
absolute_pattern_with_time = []

for time in [True, False]:
    pattern = [
        ante_num_pattern
        + day_pattern
        + d
        + month_pattern
        + d
        + year_pattern
        + (time_pattern if time else "")
        + post_num_pattern
        for d in delimiters
    ] + [
        ante_num_pattern
        + year_pattern
        + d
        + numeric_month_pattern
        + d
        + numeric_day_pattern
        + (time_pattern if time else "")
        + post_num_pattern
        for d in delimiters
    ]

    no_year_pattern = [
        day
        + raw_delimiter_with_spaces_pattern
        + month
        + (time_pattern if time else "")
        + post_num_pattern
        for day in [ante_num_pattern + numeric_day_pattern, letter_day_pattern]
        for month in [letter_month_pattern]
    ]
    no_year_pattern.extend(
        [
            ante_num_pattern
            + numeric_day_pattern
            + d
            + numeric_month_pattern
            + post_num_pattern
            for d in raw_delimiters
        ]
    )
    pattern.extend(no_year_pattern)

    no_day_pattern = [
        letter_month_pattern
        + raw_delimiter_with_spaces_pattern
        + year_pattern
        + post_num_with_letter_pattern,
        ante_num_pattern
        + lz_numeric_month_pattern
        + "/"
        + year_pattern
        + post_num_with_letter_pattern,
    ]
    pattern.extend(no_day_pattern)

    no_day_no_year_pattern = [
        letter_month_pattern,
    ]
    pattern.extend(no_day_no_year_pattern)

    full_year_pattern = ante_num_pattern + fy_pattern + post_num_pattern

    pattern.append(full_year_pattern)

    pattern = [r"(?<=" + mode_pattern + r".{,3})?" + p for p in pattern]

    if time:
        absolute_pattern_with_time[:] = pattern
    else:
        absolute_pattern[:] = pattern
