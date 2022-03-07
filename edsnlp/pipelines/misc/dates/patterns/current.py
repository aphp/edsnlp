from typing import List

from edsnlp.utils.regex import make_pattern

current_patterns: List[str] = [
    r"cette\sann[ée]e(?![-\s]l[àa])",
    r"ce\sjour",
    r"ces\sjours[-\s]ci",
    r"aujourd'?hui",
    r"ce\smois([-\s]ci)?",
    r"cette\ssemaine",
    r"cet?\s([ée]t[ée]|automne|hiver|printemps)",
]

current_pattern = make_pattern(current_patterns, with_breaks=True)
