main_pattern = dict(
    source="main",
    regex=[
        r"fibrill?ation.{1,3}(atriale|auriculaire)",
        r"flutter",
        r"brady.?arythmie",
    ],
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bFA\b",
        r"\bACFA\b",
    ],
    regex_attr="TEXT",
)

default_patterns = [
    main_pattern,
    acronym,
]
