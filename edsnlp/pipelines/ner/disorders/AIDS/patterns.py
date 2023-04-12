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
    exclude=dict(
        regex=["serologie", "prelevement"],
        window=(-20, 20),
        limit_to_sentence=False,
    ),
    assign=[
        dict(
            name="opportunist",
            regex=r"("
            + r"|".join(
                [
                    r"kapo[sz]i",
                    r"toxoplasmose",
                    r"meningo.?encephalite.toxo",
                    r"pneumocystose",
                    r"\bpep\b",
                    r"pneumocystis",
                    r"cryptococcose",
                    r"cytomégalovirus",
                    r"myobact",
                    r"opportunist",
                    r"co.?infect",
                ]
            )
            + ")"
            + r"(?!.{0,20}(?:non|0))",
            window=(-10, 30),
            limit_to_sentence=False,
        ),
        dict(
            name="stage",
            regex=r"stade.{0,5}\b(b|c)\b",
            window=10,
        ),
    ],
    regex_attr="NORM",
)

default_patterns = [
    aids,
    hiv,
]
