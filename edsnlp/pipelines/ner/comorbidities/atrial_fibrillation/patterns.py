main_pattern = dict(
    source="main",
    regex=[
        r"\bfa\b",
        r"\bacfa\b",
        r"fibrill?ation.{1,3}(atriale|auriculaire)",
        r"flutter",
        r"brady.?cardie",
        r"brady.?arythmie",
    ],
)

default_patterns = [
    main_pattern,
]
