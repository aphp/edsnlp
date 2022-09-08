default_patterns = dict(
    source="alcohol",
    regex=[
        r"\balco[ol]",
        r"\bethyl",
        r"(?<!25-)\boh\b",
    ],
    exclude=dict(
        regex=[
            "occasion",
            "moder",
        ],
        window=(0, 5),
    ),
    regex_attr="NORM",
    assign=[
        dict(
            name="verres",
            regex=r"(\d+[\.,-]?\d*)\s*verre [^\\d,;:]*$'",
            window=(-3, 5),
        ),
        dict(
            name="grammes",
            regex=r"(\d+[\.,-]?\d*)\s*(?:g|grammes?)\s*(?:/|par)\s*j",
            window=(-3, 5),
        ),
        dict(
            name="sevrage",
            regex=r"(?<!non )(?<!pas )(sevr|arret|stop|ancien)",
            window=(-5, 5),
        ),
        dict(
            name="zero_after",
            regex=r"^[a-z]*\s*:?\s*(0|oui|non(?! sevr))",
            window=6,
        ),
    ],
)
