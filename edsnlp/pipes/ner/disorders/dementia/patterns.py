main_pattern = dict(
    source="main",
    regex=[
        r"demence",
        r"demense",
        r"dementiel",
        r"corps\s*de\s*le[vw]y",
        r"deficits?.chroniques?.cognitifs?",
        r"troubles?.mnesique?",
        r"troubles?.praxique",
        r"troubles?.att?entionel",
        r"troubles?.degeneratifs?.{1,15}fonctions.{1,5}sup",
        r"maladies?.cerebrales?.degen",
        r"troubles?.neurocogn\w+",
        r"deficits?.cogniti\w+",
        r"atteinte.{1,7}spheres?cogniti",
        r"syndrome?.{1,10}(frontal|neuro.deg)",
        r"(trouble|d(y|i)sfonction).{1,25}cogni\w+",
        r"(?<!specialisee)alzheimer",
        r"demence.{1,20}(\balz|\bpark)",
        r"binswanger",
        r"gehring",
        r"\bpick",
        r"de\s*guam",
        r"[kc]reutzfeld.{1,5}ja[ck]ob",
        r"huntington",
        r"korsako[fv]",
        r"atrophie.{1,10}(cortico|hip?pocamp|cereb|lobe)",
    ],
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bSLA\b",
        r"\bDFT\b",
        r"\bDFT",
        r"\bTNC\b",
        r"\bHTT\b",
        r"\bALS\b",
    ],
    regex_attr="TEXT",
    exclude=dict(
        regex=r"\banti",  # anticorps
        window=-15,
        regex_attr="NORM",
    ),
)

charcot = dict(
    source="charcot",
    regex=[
        r"maladie.{1,10}charcot",
        r"maladie.{1,10}lou\s*gehrig",
    ],
    exclude=dict(
        regex=[
            "pied de",
            "marie.?tooth",
        ],
        window=(-3, 3),
    ),
)


default_patterns = [
    main_pattern,
    acronym,
    charcot,
]
