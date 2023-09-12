PA = r"(?:\bp/?a\b|paquets?.?annee)"
QUANTITY = r"(?P<quantity>[\d]{1,3})"
PUNCT = r"\.,-;\(\)"

default_patterns = [
    dict(
        source="tobacco",
        regex=[
            r"tabagi",
            r"tabac",
            r"\bfume\b",
            r"\bfumeu",
            r"\bpipes?\b",
        ],
        exclude=dict(
            regex=[
                "occasion",
                "moder",
                "quelqu",
                "festi",
                "rare",
                "sujet",  # Example : Chez le sujet fumeur ... generic sentences
            ],
            window=(-3, 5),
        ),
        regex_attr="NORM",
        assign=[
            dict(
                name="stopped",
                regex=r"(?<!non )(?<!pas )(\bex\b|sevr|arret|stop|ancien)",
                window=(-3, 15),
            ),
            dict(
                name="zero_after",
                regex=r"^[a-z]*\s*:?[\s-]*(0|non(?! sevr))",
                window=6,
            ),
            dict(
                name="PA",
                regex=rf"{QUANTITY}[^{PUNCT}]{{0,10}}{PA}|{PA}[^{PUNCT}]{{0,10}}{QUANTITY}",
                window=(-10, 10),
                reduce_mode="keep_first",
            ),
            dict(
                name="secondhand",
                regex="(passif)",
                window=5,
                reduce_mode="keep_first",
            ),
        ],
    )
]
