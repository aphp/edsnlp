default_patterns = dict(
    source="alcohol",
    regex=[
        r"\balco[ol]",
        r"\bethyl",
        r"(?<!25.?)\boh\b",
        r"exogenose",
    ],
    exclude=[
        dict(
            regex=[
                "occasion",
                "moder",
                "quelqu",
                "festi",
                "rare",
                "libre",  # OH-libres
            ],
            window=(0, 5),
        ),
        dict(
            regex="pansement",
            window=-3,
        ),
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="stopped",
            regex=r"(?<!non )(?<!pas )(sevr|arret|stop|ancien)",
            window=(-5, 5),
        ),
        dict(
            name="zero_after",
            regex=r"^[a-z]*\s*:?[\s-]*(0|oui|non(?! sevr))",
            window=6,
        ),
    ],
)
