from ..terms import ASYMPTOMATIC, BRAIN, HEART, PERIPHERAL

acronym = dict(
    source="acronym",
    regex=[
        r"\bAOMI\b",
        r"\bACOM\b",
        r"\bTAO\b",
        r"\bSAPL\b",
        r"\bOACR\b",
        r"\bOVCR\b",
        r"\bSCS\b",
        r"\bTVP\b",
        r"\bCAPS\b",
        r"\bMTEV\b",
        r"\bPTT\b",
        r"\bMAT\b",
        r"\bSHU\b",
    ],
    regex_attr="TEXT",
)

other = dict(
    source="other",
    regex=[
        r"\bbuerger",
        r"takayasu",
        r"\bhorton",
        r"wegener",
        r"churg.{1,10}strauss",
        r"\bsneddon",
        r"budd.chiari",
        r"infarctus.{1,5}(renal|splenique|polaire|pulmo)",
        r"ulcere.{1,5}arter",
        r"syndrome.?hemolytique.{1,8}uremique",
        r"granulomatose.{1,10}polyangeite",
        r"occlusion.{1,10}(artere|veine).{1,20}retine",
        r"syndrome.{1,20}anti.?phospho",
        r"embolie.{1,5}pulm",
    ],
    regex_attr="NORM",
)

with_localization = dict(
    source="with_localization",
    regex=[
        r"angiopathie",
        r"arteriopathies.{1,5}obliterante",
        r"gangren",
        r"claudication",
        r"dissection.{1,10}(aort|arter)",
        r"tromboangeit",
        r"tromboarterit",
        r"(pontage|angioplastie).{1,10}(\bfem|\bpop|\bren|\bjamb)",
        r"arterite",
        r"(ischemie|infarctus).{1,10}mesenterique",
        r"endarteriectomie",
        r"vascularite",
        r"occlusion.{1,10}terminaisons? carotid",
        r"cryoglobulinemie",
        r"colite.{1,5}ischemi",
        r"embole.{1,10}cholesterol",
        r"purpura.?thrombopenique.?idiopa",
        r"micro.?angiopathie.?thrombotique",
    ],
    exclude=[
        dict(
            regex=BRAIN + HEART + ASYMPTOMATIC + [r"inr\srecommande\ssous\savk"],
            window=(-8, 8),
            limit_to_sentence=False,
        ),
    ],
    regex_attr="NORM",
)

thrombosis = dict(
    source="thrombosis",
    regex=[
        r"thrombos",
        r"thrombol[^y]",
        r"thrombophi",
        r"thrombi[^n]",
        r"thrombus",
        r"thrombectomi",
        r"thrombo.?embo",
        r"phlebit",
    ],
    exclude=[
        dict(
            regex=BRAIN + HEART + ["superficiel", "\biv\b", "intra.?vein"],
            window=(-15, 15),
            limit_to_sentence=False,
        ),
        dict(
            regex=[
                "pre",
                "anti",
                "bilan",
            ],
            window=-4,
        ),
    ],
    regex_attr="NORM",
)


ischemia = dict(
    source="ischemia",
    regex=[
        r"ischemi",
    ],
    exclude=[
        dict(
            regex=BRAIN + HEART,
            window=(-7, 7),
        ),
    ],
    assign=[
        dict(
            name="peripheral",
            regex="(" + r"|".join(PERIPHERAL) + ")",
            window=15,
        ),
    ],
    regex_attr="NORM",
)

ep = dict(
    source="ep",
    regex=r"\bEP(?![\w\./-])",
    regex_attr="TEXT",
    exclude=[
        dict(
            regex=[
                r"fibreux",
                r"retin",
                r"\bfove",
                r"\boct\b",
                r"\bmacula",
                r"prosta",
                r"\bip\b",
                r"protocole",
                r"seance",
                r"echange",
                r"ritux",
                r"ivig",
                r"ig.?iv",
                r"\bctc",
                r"corticoide",
                r"serum",
                r"\bcure",
                r"plasma",
                r"mensuel",
                r"semaine",
                r"serologi",
                r"espaces.porte",
                r"projet",
                r"bolus",
            ],
            window=(-25, 25),
            limit_to_sentence=False,
            regex_attr="NORM",
        ),
        dict(
            regex=[r"rdv", r"les", r"des", r"angine"],
            window=(-3, 0),
            regex_attr="NORM",
        ),
    ],
)

hypertension = dict(
    source="main",
    regex=[
        r"\bhta\b",
        r"hyper.?tension.?arte",
        r"hyper.?tendu",
        r"hyper.?tension.?essenti",
        r"hypertensi",
    ],
    exclude=dict(
        regex="(pulmo|porta)",
        window=3,
    ),
)

default_patterns = [
    acronym,
    other,
    with_localization,
    thrombosis,
    ep,
    ischemia,
    hypertension,
]
