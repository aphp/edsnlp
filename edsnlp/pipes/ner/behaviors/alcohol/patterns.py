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
            regex=r"(\bex\b|sevr|arret|stop|ancien)",
            window=(-3, 15),
            reduce_mode="keep_first",
        ),
        dict(
            name="zero_after",
            regex=r"(?=^[a-z]*\s*:?[\s-]*(0|non|aucun|jamais))",
            window=3,
            reduce_mode="keep_first",
        ),
    ],
)
