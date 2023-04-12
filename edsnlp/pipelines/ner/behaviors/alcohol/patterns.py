default_patterns = dict(
    source="alcohol",
    regex=[
        r"\balco[ol]",
        r"\bethyl",
        r"(?<!(25.?)|(sevrage)).?\boh\b",
        r"exogenose",
        r"delirium.tremens",
    ],
    exclude=[
        dict(
            regex=[
                "occasion",
                "episod",
                "festi",
                "rare",
                "libre",  # OH-libres
                "aigu",
            ],
            window=(-3, 5),
        ),
        dict(
            regex=["pansement", "compress"],
            window=-3,
        ),
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="stopped",
            regex=r"(?<!non )(?<!pas )(sevr|arret|stop|ancien)",
            window=(-3, 5),
        ),
        dict(
            name="zero_after",
            regex=r"^[a-z]*\s*:?[\s-]*(0|oui|non(?! sevr))",
            window=6,
        ),
    ],
)
