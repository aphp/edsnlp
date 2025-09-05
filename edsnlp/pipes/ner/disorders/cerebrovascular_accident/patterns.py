import re

from edsnlp.utils.resources import get_AVC_care_site

from ..terms import BRAIN, HEART, PERIPHERAL

AVC_CARE_SITES_REGEX = [
    r"\b" + re.escape(cs.strip()) + r"\b" for cs in get_AVC_care_site(prefix=True)
] + [
    r"h[oô]p",
    r"\brcp",
    r"service",
    r"\bsau",
    r"ap.?hp",
    r"\burg",
    r"finess",
    r"\bsiret",
    r"[àa] avc",
    r"consult",
]

avc = dict(
    source="avc",
    regex=[
        r"\bavc\b",
    ],
    exclude=[
        dict(
            regex=AVC_CARE_SITES_REGEX,
            window=(-5, 5),
            regex_flags=re.S | re.I,
            limit_to_sentence=False,
        ),
        dict(
            regex=r"\b[a-z]\.",
            window=2,
            limit_to_sentence=False,
        ),
    ],
    regex_attr="NORM",
)

with_localization = dict(
    source="with_localization",
    regex=[
        r"(hemorr?agie|hematome)",
        r"angiopath",
        r"angioplasti",
        r"infarctus",
        r"occlusion",
        r"saignement",
        r"embol",
        r"vascularite",
        r"\bhsd\b",
        r"thrombos",
        r"thrombol[^y]",
        r"thrombophi",
        r"thrombi[^n]",
        r"thrombus",
        r"thrombectomi",
        r"phleb",
    ],
    regex_attr="NORM",
    exclude=[
        dict(
            regex=r"pulmo|poumon",
            window=4,
        ),
    ],
    assign=[
        dict(
            name="brain_localized",
            regex="(" + r"|".join(BRAIN) + ")",
            window=(-15, 15),
            limit_to_sentence=False,
            include_assigned=False,
        ),
    ],
)

general = dict(
    source="general",
    regex=[
        r"accident.{1,5}vasculaires.{1,5}cereb",
        r"accident.{1,5}vasculaire.{1,5}ischemi",
        r"accident.{1,5}ischemi",
        r"moya.?moya",
        r"occlusion.{1,5}(artere|veine).{1,20}retine",
        r"vasculopathies?.cerebrales?.ischemique",
        r"maladies?.des.petites.arteres",
        r"maladies?.des.petits.vaisseaux",
        r"thrombolyse",
        r"\bsusac\b",
    ],
    regex_attr="NORM",
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bAIC\b",
        r"\bOACR\b",
        r"\bOVCR\b",
    ],
    regex_attr="TEXT",
)

AIT = dict(
    source="AIT",
    regex=[
        r"\bAIC\b",
        r"\bOACR\b",
        r"\bOVCR\b",
        r"\bAIT\b",
    ],
    regex_attr="TEXT",
)

ischemia = dict(
    source="ischemia",
    regex=[
        r"ischemi",
    ],
    exclude=[
        dict(
            regex=PERIPHERAL + HEART,
            window=(-7, 7),
        ),
    ],
    assign=[
        dict(
            name="brain",
            regex="(" + r"|".join(BRAIN) + ")",
            window=(-10, 15),
        ),
    ],
    regex_attr="NORM",
)

default_patterns = [
    avc,
    with_localization,
    general,
    acronym,
    AIT,
    ischemia,
]
