import re

from edsnlp.utils.resources import get_AVC_care_site_prefixes

from ..terms import BRAIN

AVC_CARE_SITES_REGEX = [
    r"\b" + re.escape(cs.strip()) + r"\b" for cs in get_AVC_care_site_prefixes()
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
        ),
        dict(
            regex=r"\b[a-z]\.",
            window=2,
        ),
    ],
    regex_attr="NORM",
)

with_localization = dict(
    source="with_localization",
    regex=[
        r"(hemorr?agie|hematome)",
        r"infarctus",
        r"occlusion",
        r"saignement",
        r"thrombo(se|phlebite)",
        r"embol.{1,10}",
        r"vascularite",
        r"\bhsd\b",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="brain_localized",
            regex="(" + r"|".join(BRAIN) + ")",
            window=8,
        ),
    ],
)

general = dict(
    source="general",
    regex=[
        r"accidents? vasculaires? cereb",
        r"accidents? vasculaires? ischemi",
        r"accidents? ischemi",
        r"moya.?moya",
        r"occlusion.{1,5}(artere|veine).{1,20}retine",
        r"vasculopathies?.cerebrales?.ischemique",
        r"maladies? des petites arteres",
        r"maladies? des petits vaisseaux",
        r"thrombolyse.{1,10}(iv|intra.?vein)",
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

default_patterns = [
    avc,
    with_localization,
    general,
    acronym,
    AIT,
]
