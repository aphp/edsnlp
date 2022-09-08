main_pattern = dict(
    source="main",
    regex=[
        r"\bds?n?id\b",
        r"\bdiabet[^o]",
    ],
    exclude=dict(
        regex=[
            "insipide",
            "nephrogenique",
            "aigu",
            r"\bdr\b",  # Dr. ...
            "endocrino",  # Section title
            "cortico",
            "soins aux pieds",  # Section title
            "nutrition",  # Section title
            r"\s?:\n+\W+(?!oui|non|\W)",  # General pattern for section title
        ],
        window=(-5, 5),
    ),
    regex_attr="NORM",
    assign=[
        dict(
            name="complicated_before",
            regex="("
            + r"|".join(
                [
                    r"nephropat",
                    r"neuropat",
                    r"retinopat",
                    r"glomerulopathi",
                    r"neuroangiopathi",
                ]
            )
            + ")",
            window=-3,
        ),
        dict(
            name="complicated_after",
            regex="("
            + r"|".join(
                [
                    r"(?<!sans )compli",
                    r"(?<!a)symptomatique",
                ]
            )
            + ")",
            window=7,
        ),
        dict(
            name="type",
            regex=r"type.(i|ii|1|2)",
            window=6,
        ),
        dict(
            name="insulin",
            regex=r"insulino.?(dep|req)",
            window=6,
        ),
    ],
)

complicated_pattern = dict(
    source="complicated",
    regex=[
        r"mal perforant plantaire",
        r"pieds? diabeti",
    ],
    exclude=dict(
        regex="soins aux",  # Section title
        window=-2,
    ),
    regex_attr="NORM",
)

default_patterns = [
    main_pattern,
    complicated_pattern,
]
