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

default_patterns = [altered, other]
