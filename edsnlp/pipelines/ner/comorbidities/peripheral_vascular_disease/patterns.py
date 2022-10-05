from ..terms import BRAIN, HEART, ASYMPTOMATIC

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
        r"infarctus.{1,5}(renal|splenique|polaire)",
    ],
    regex_attr="NORM",
)

with_localization = dict(
    source="with_localization",
    regex=[
        r"angiopathie",
        r"arteriopathies? obliterante",
        r"gangren",
        r"claudication",
        r"dissection.{1,10}(aort|arter)",
        r"tromboangeit",
        r"tromboarterit",
        r"pontage.{1,10}(fem|pop)?",
        r"arterite",
        r"(ischemie|infarctus).{1,10}mesenterique",
        r"endarteriectomie",
        r"vascularite",
        r"granulomatose.{1,10}polyangeite",
        r"occlusion.{1,10}(artere|veine).{1,20}retine",
        r"syndrome.{1,20}anti.?phospho",
        r"occlusion.{1,10}terminaisons? carotid",
        r"embolies? pulmo",
        r"cryoglobulinemie",
        r"colites? ischemi",
        r"embole.{1,10}cholesterol",
        r"purpura.?thrombopenique.?idiopa",
        r"micro.?angiopathie.?thrombotique",
        r"syndrome.?hemolytique.{1,8}uremique",
    ],
    exclude=[
        dict(
            regex=BRAIN + HEART,
            window=8,
        ),
    ],
    regex_attr="NORM",
)

thrombosis = dict(
    source="thrombosis",
    regex=[
        r"thrombos",
        r"thrombol",
        r"thrombi[^n]",
        r"thrombus",
    ],
    exclude=[
        dict(
            regex=BRAIN + HEART + ["superficiel", "iv", "intra.?vein"],
            window=4,
        ),
        dict(
            regex=[
                "pre",
                "anti",
            ],
            window=-4,
        ),
    ],
    regex_attr="NORM",
)

default_patterns = [
    acronym,
    other,
    with_localization,
    thrombosis,
]
