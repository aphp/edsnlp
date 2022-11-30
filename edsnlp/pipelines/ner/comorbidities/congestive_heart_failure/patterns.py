from ..terms import ASYMPTOMATIC

main_pattern = dict(
    source="main",
    regex=[
        r"defaillance.{1,10}cardi",
        r"(œ|oe)deme.{1,10}pulmon",
        r"(œ|oe)deme.{1,10}poumon",
        r"decompensation.{1,10}card",
        r"chocs?\s(septi.{1,10}|cardio)",
        r"greffe.{1,10}c(œ|oe)ur",
        r"greffe.{1,10}cardia",
        r"transplantation.{1,10}c(œ|oe)ur",
        r"transplantation.{1,10}cardia",
        r"arret.{1,10}cardi",
        r"c(œ|oe)ur pulmo",
        r"foie card",
    ],
    regex_attr="NORM",
)

symptomatic = dict(
    source="symptomatic",
    regex=[
        r"cardiopathi",
        r"cardiomyopathi",
        r"insuffisance.{1,10}(cardi|diasto|ventri)",
        r"d(i|y)sfonction.{1,15}(ventricul|\bvg|cardiaque)",
    ],
    regex_attr="NORM",
    exclude=dict(
        regex=ASYMPTOMATIC + ["ischemi"],  # Exclusion of ischemic events
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
        r"\bACFA\b",
    ],
    regex_attr="TEXT",
)

default_patterns = [
    main_pattern,
    symptomatic,
    acronym,
    AF_main_pattern,
    AF_acronym,
]
