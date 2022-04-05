from typing import List

from edsnlp.utils.regex import make_pattern

current_patterns: List[str] = [
    r"(?P<year>cette\s+ann[ée]e)(?![-\s]l[àa])",
    r"(?P<day>ce\s+jour|aujourd['\s]?hui)",
    r"(?P<week>cette\s+semaine|ces\sjours[-\s]ci)",
    r"(?P<month>ce\smois([-\s]ci)?)",
    r"(?P<season_summer>cet\s+[ée]t[ée])",
    r"(?P<season_fall>cet\s+automne)",
    r"(?P<season_winter>cet\s+hiver)",
    r"(?P<season_spring>ce\s+printemps)",
]

current_pattern = make_pattern(current_patterns, with_breaks=True)
