from ..utils import normalize_space_characters

altered = dict(
    source="altered",
    regex=[
        "iatrogenie",
        "polymediqu",
        "polymedication",
    ],
    regex_attr="NORM",
)
other = dict(
    source="other",
    regex=[
        r"conciliation[\s]+medicament",
    ],
    regex_attr="NORM",
)

default_patterns = normalize_space_characters([altered, other])
