main_pattern = dict(
    source="main",
    regex=[
        r"\bhta\b",
        r"hyper.?tension.?arte",
        r"hyper.?tendu",
    ],
    exclude=dict(
        regex="pulmo",
        window=3,
    ),
)

default_patterns = [
    main_pattern,
]
