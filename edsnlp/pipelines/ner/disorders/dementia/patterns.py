main_pattern = dict(
    source="main",
    regex=[
        r"demence",
        r"dementiel",
        r"corps de le[vw]y",
        r"deficits?.chroniques?.cognitif",
        r"troubles?.mnesique?",
        r"troubles?.praxique",
        r"troubles?.attentionel",
        r"troubles?.degeneratif.{1,15}fonctions.{1,5}sup",
        r"maladies?.cerebrales?.degen",
        r"troubles?.neurocogn",
        r"deficits?.cognitif",
        r"(trouble|dysfonction).{1,20} cogniti",
        r"atteinte.{1,7}spheres?cogniti",
        r"syndrome.{1,10}(frontal|neuro.deg)",
        r"dysfonction.{1,25}cogni",
        r"(?<!specialisee )alzheimer",
        r"demence.{1,20}(\balz|\bpark)",
        r"binswanger",
        r"gehring",
        r"\bpick",
        r"de guam",
        r"[kc]reutzfeld.{1,5}ja[ck]ob",
        r"huntington",
        r"korsako[fv]",
        r"atrophie.{1,10}(cortico|hippocamp|cereb|lobe)",
    ],
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bSLA\b",
        r"\bDFT\b",
        r"\bDFT",
        r"\bTNC\b",
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
