from typing import Optional

from ..patterns.directions import directions
from ..patterns.numbers import letter_numbers as letter_numbers_patterns
from ..patterns.relative import specific
from .factory import str2int, time2int_factory, time2int_fast_factory
from .numbers import letter_numbers

number2int_fast = time2int_fast_factory(letter_numbers)
number2int = time2int_factory(letter_numbers_patterns)


def parse_number(snippet: str) -> Optional[int]:
    return str2int(snippet) or number2int_fast(snippet) or number2int(snippet)


parse_specific = time2int_factory(specific)

parse_direction = time2int_factory(directions, use_as_default=True)
