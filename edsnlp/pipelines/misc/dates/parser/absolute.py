from typing import Optional

from edsnlp.pipelines.misc.dates.patterns.atomic import days, months

from .factory import str2int, time2int_factory, time2int_fast_factory

# Days
day2int_fast = time2int_fast_factory(days.letter_days_dict_simple)
day2int = time2int_factory(days.letter_days_dict)

# Months
month2int_fast = time2int_fast_factory(months.letter_months_dict_simple)
month2int = time2int_factory(months.letter_months_dict)


def parse_month(snippet: str) -> Optional[int]:
    return str2int(snippet) or month2int_fast(snippet) or month2int(snippet)


def parse_day(snippet: str) -> Optional[int]:
    return str2int(snippet) or day2int_fast(snippet) or day2int(snippet)
