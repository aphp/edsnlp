TO_EXCLUDE = r"(?<!a )((\bacc\b)|anti.?coag|anti.?corps|buschke|(\bac\b)|(\bbio))"

main_pattern = dict(
    source="main",
    regex=[
        r"arth?rites?.{1,5}juveniles?.{1,5}idiopa\w+",
        r"myosite",
        r"myopath\w+.{1,5}inflammatoire",
        r"polyarth?rite.{1,5}chroni\w+.{1,5}evol",
        r"polymyosie",
        r"polyarth?rites?.{1,5}(rhizo|rhuma)",
        r"scleroderm\w+",
        r"connectivite",
        r"sarcoidose",
    ],
    exclude=dict(
        regex=[TO_EXCLUDE],
        window=(-7, 7),
    ),
    regex_attr="NORM",
)

lupus = dict(
    source="lupus",
    regex=[
        r"\blupus",
    ],
    regex_attr="NORM",
)

lupique = dict(
    source="lupique",
    regex=[r"\blupique", r"\blupic"],
    exclude=dict(
        regex=[TO_EXCLUDE],
        window=(-7, 7),
    ),
    regex_attr="NORM",
)

acronym = dict(
    source="acronyms",
    regex=[
        r"\bAJI\b",
        r"\bLED\b",
        r"\bPCE\b",
        r"\bCREST\b",
        r"\bPPR\b",
        r"\bMICI\b",
        r"\bMNAI\b",
    ],
    regex_attr="TEXT",
)

named_disease = dict(
    source="named_disease",
    regex=[
        r"libman.?lack",
        r"\bstill",
        r"felty",
        r"forestier.?certon",
        r"gou(g|j)erot",
        r"raynaud",
        r"thibierge.?weiss",
        r"sjogren",
        r"gou(g|j)erot.?sjogren",
    ],
    regex_attr="NORM",
)

default_patterns = [
    main_pattern,
    lupus,
    lupique,
    acronym,
    named_disease,
]
