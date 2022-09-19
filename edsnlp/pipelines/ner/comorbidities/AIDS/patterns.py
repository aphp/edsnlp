aids = dict(
    source="aids",
    regex=[
        r"(vih.{1,5}stade.{1,5})?\bsida\b",
    ],
    regex_attr="NORM",
)

hiv = dict(
    source="hiv",
    regex=[
        r"\bhiv\b",
        r"\bvih\b",
    ],
    assign=dict(
        name="opportunist",
        regex=r"("
        + r"|".join(
            [
                r"kapo[sz]i",
                r"toxoplasmose",
                r"\bmet\b",
                r"meningo.?encephalite toxo",
                r"pneumocystose",
                r"\bpep\b",
                r"pneumocystis",
                r"cryptococcose",
                r"cytomégalovirus",
                r"\bcmv\b",
                r"myobact",
                r"opportunist",
            ]
        )
        + ")",
        window=(-8, 8),
    ),
    regex_attr="NORM",
)

default_patterns = [
    aids,
    hiv,
]
