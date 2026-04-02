from ..utils import normalize_space_characters

healthy = dict(
    source="healthy",
    regex=[
        r"\bcontinen(te?|ce)",
    ],
    regex_attr="NORM",
)
severe = dict(
    source="severe",
    regex=[
        r"\bsad\b",
        "sonde a demeure",
        "sonde urinaire",
        r"sonde vesicale?",
        "sondage urinaire",
        "sondage vesical",
        r"sondages? allers?[\s-]retours?",
    ],
    regex_attr="NORM",
)
altered = dict(
    source="altered",
    regex=[
        r"fuites? (anales?|urinaires?)",
        r"incontinente?",
        "incontinence",
        r"pollakiurie nocturne",
        "urgenterie",
        r"globe (?:urinaire|vesicale?)",
        r"troubles? sphincteriens?",
        r"incontinence(?: urinaire| anale)?",
        r"signe fonctionnel urinaire",
        r"dysurie",
        r"retention d'urines?",
        r"\bsfu\b",
        "miction difficile",
    ],
    regex_attr="NORM",
    exclude=dict(
        regex="aigue?s?",
        window=(-3, 3),
    ),
)
severe_lower = dict(source="severe_lower", regex=[r"\bcouches?\b"], regex_attr="LOWER")
protection = dict(
    source="other_protection",
    regex=["protection"],
    exclude=dict(
        regex=[
            "juridique",
            "donnees",
            "tutelle",
            "curatelle",
        ],
        window=5,
    ),
)

lever_nocturne = dict(
    source="mild_noturnal",
    regex=["lever nocturne"],
    regex_attr="NORM",
    assign=dict(name="urinate", regex="(uriner)", window=5, required=True),
)

other = dict(
    source="other",
    regex=[
        "miction spontanee",
        r"reeducation vesicale",
        r"chaise percee",
    ],
    regex_attr="NORM",
)

default_patterns = normalize_space_characters(
    [
        healthy,
        altered,
        severe,
        other,
        severe_lower,
        protection,
        lever_nocturne,
    ]
)
