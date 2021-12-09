from datetime import date
from typing import List

from ..utils import make_pattern

year_patterns: List[str] = [
    r"19\d\d",
    r"\d\d",
] + [str(year) for year in range(2000, date.today().year + 2)]

year_pattern = make_pattern(year_patterns)
