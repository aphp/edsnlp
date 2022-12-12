TO_EXCLUDE = r"(?<!a )((\bacc\b)|anti.?coag|anti.?corps|buschke|(\bac\b))"

main_pattern = dict(
    source="main",
    regex=[
        r"arthrites? juveniles? idiopa",
        r"myosite",
        r"myopathie.{1,5}inflammatoire",
        r"lupus",
        r"lupique",
        r"polyarthrite chronique evol",
        r"polymyosie",
        r"polyarthrites? (rhizo|rhuma)",
        r"sclerodermie",
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
        r"lupique",
    ],
    exclude=dict(
        regex=[TO_EXCLUDE],
        window=(-7, 7),
    ),
    regex_attr="NORM",
)

acronym = dict(
    source="complicated",
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
    acronym,
    named_disease,
]
