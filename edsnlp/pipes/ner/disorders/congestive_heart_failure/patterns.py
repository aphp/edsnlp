from ..terms import ASYMPTOMATIC

main_pattern = dict(
    source="main",
    regex=[
        r"defaill?ance.{1,10}cardi\w+",
        r"(œ|oe)deme?.{1,10}pulmon",
        r"decompensation.{1,10}card",
        r"choc.{1,30}cardio",
        r"greff?e.{1,10}c(œ|oe)ur",
        r"greff?e.{1,10}cardia",
        r"transplantation.{1,10}c(œ|oe)ur",
        r"transplantation.{1,10}cardia",
        r"arret.{1,10}cardi",
        r"c(œ|oe)ur pulmo",
        r"foie.card",
        r"pace.?maker",
        r"stimulateur.cardiaque",
        r"valve.{1,30}(meca|artific)",
    ],
    regex_attr="NORM",
)

symptomatic = dict(
    source="symptomatic",
    regex=[
        r"cardio\s*path\w+",
        r"cardio\s*myopath\w+",
        r"d(i|y)sfonction.{1,15}(ventricul|\bvg|cardiaque)",
        r"valvulo\s*path\w+?",
        r"\bic\b.{1,10}(droite|gauche)",
    ],
    regex_attr="NORM",
    exclude=dict(
        regex=ASYMPTOMATIC + [r"(?<!\bnon.)ischem"],  # Exclusion of ischemic events
        window=5,
    ),
)

with_minimum_severity = dict(
    source="min_severity",
    regex=[
        r"insuf?fisance.{1,10}(\bcardi|\bdiasto|\bventri|\bmitral|tri.?cusp)",
        r"(retrecissement|stenose).(aortique|mitral)",
        r"\brac\b",
        r"\brm\b",
    ],
    regex_attr="NORM",
    exclude=dict(
        regex=ASYMPTOMATIC + ["minime", "modere", r"non.serre"],
        window=5,
    ),
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bOAP\b",
        r"\bCMH\b",
    ],
    regex_attr="TEXT",
)

AF_main_pattern = dict(
    source="AF_main",
    regex=[
        r"fibrill?ation.{1,3}(atriale|auriculaire|ventriculaire)",
        r"flutter",
        r"brady.?arythmie",
        r"pace.?maker",
    ],
)

AF_acronym = dict(
    source="AF_acronym",
    regex=[
        r"\bFA\b",
        r"\bAC.?FA\b",
    ],
    regex_attr="TEXT",
)

default_patterns = [
    main_pattern,
    symptomatic,
    acronym,
    AF_main_pattern,
    AF_acronym,
    with_minimum_severity,
]
