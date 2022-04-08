from typing import List

from edsnlp.utils.regex import make_pattern

current_patterns: List[str] = [
    r"(?P<year_0>cette\s+ann[ée]e)(?![-\s]l[àa])",
    r"(?P<day_0>ce\s+jour|aujourd['\s]?hui)",
    r"(?P<week_0>cette\s+semaine|ces\sjours[-\s]ci)",
    r"(?P<month_0>ce\smois([-\s]ci)?)",
]

current_pattern = make_pattern(current_patterns, with_breaks=True)
