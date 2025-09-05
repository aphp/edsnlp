from ..terms import HEART

main_pattern = dict(
    source="main",
    regex=[
        r"coronaropathie",
        r"angor.{1,5}instable",
        r"cardiopathie(?!.{0,20}non).{0,20}(ischem|arteriosc)",
        r"cardio.?myopathie(?!.{0,20}non).{0,20}(ischem|arteriosc)",
        r"ischemi.{1,15}myocard",
        r"syndrome.{1,5}corona.{1,10}aigu",
        r"syndrome.{1,5}corona.{1,10}st",
        r"pontage.{1,5}mammaire",
    ],
    regex_attr="NORM",
)

with_localization = dict(
    source="with_localization",
    regex=[
        r"\bstent",
        r"endoprothese",
        r"pontage",
        r"anevr[iy]sme",
        "infarctus",
        r"angioplasti",
    ],
    assign=[
        dict(
            name="heart_localized",
            regex="(" + r"|".join(HEART) + ")",
            window=(-10, 10),
        ),
    ],
    regex_attr="NORM",
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bidm\b",
        r"\bsca\b",
        r"\batl\b",
    ],
    regex_attr="NORM",
    assign=dict(
        name="segment",
        regex=r"st([+-])",
        window=2,
    ),
)


default_patterns = [
    main_pattern,
    with_localization,
    acronym,
]
