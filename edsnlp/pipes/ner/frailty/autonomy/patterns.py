from ..utils import make_assign_regex, make_status_assign, normalize_space_characters

severe = dict(
    source="severe",
    regex=[
        r"maintien a domicile difficile",
        "aide totale",
        "ehpad",
        r"\bmadd?\b",
        "mise au fauteuil",
    ],
    regex_attr="NORM",
)

mild = dict(
    source="mild",
    regex=[
        r"incapacite a revenir chez (?:elle|lui)",
        "sort de moins en moins",
        r"aux(iliaires?)? de (?:vie|soins?)",
        r"\badv\b",
        "portage des? repas",
        "aides? professionnelles?",
        "aides? exterieure?s?",
        r"ne sort (?:plus|pas)",
        "chaise percee",
        r"accompagnee? aux toilettes",
        r"aides? au? domicile",
    ],
    regex_attr="NORM",
)


other = dict(
    source="other",
    regex=[
        r"\bautonomie",
        "(?:sur le )?plan fonctionnel",
        "(?:sur le )?plan de l'autonomie",
        "etat fonctionnel",
        "conduit sa voiture",
        "sans aide professionnelle",
        "sans aide exterieure",
        "vit a domicile",
        "sort seule?",
        r"conduisait jusqu'a (?:peu|recemment)",
        r"\bdependence",
        r"\bautonome\b",
        r"recuperation de l'autonomie( initiale)?",
        "perte d'autonomie",
        r"utilise le telephone(?: portable)?",
        r"re[\s-]?autonomisation",
        r"ergotherap(?:ie|eute|eutique)",
    ],
    regex_attr="NORM",
    assign=make_status_assign(-4, 4),
)


other_orth = dict(source="other_orth", regex=["SSR"], regex_attr="ORTH")
severe_orth = dict(source="severe_orth", regex=[r"\bEPHAD\b"], regex_attr="ORTH")

house_stay = dict(
    source="other_stay",
    regex=[
        "maintien au? domicile",
        r"\bmad\b",
    ],
    regex_attr="NORM",
    assign=dict(
        name="severe",
        regex=make_assign_regex(["difficile", "difficulte"]),
        window=(-2, 2),
    ),
)

FAMILY_COMPLEMENTS = ["fils", "fille", "enfants", r"epou(?:x|se)", "mari", "femme"]
ALTERED_COMPLEMENTS = [
    r"\baidee?s?\b",
    rf"aidee? (?:par (?:la|les?|l' ?)?)?(?:{'|'.join(member for member in FAMILY_COMPLEMENTS)})",  # noqa E501
    "mauvaise?",
    "supervision",
]
SEVERE_COMPLEMENTS = [
    "dependante?",
    rf"(?:geres?)? (?:par (?:la|les?|l' ?)?)?(?:{'|'.join(member for member in FAMILY_COMPLEMENTS)})",  # noqa E501
]
HEALTHY_COMPLEMENTS = [
    "autonome",
    r"sans\saide",
    "bon(?:ne)?(?! observance)",
    "maintenir",
    "preserver",
    "(?<!paracetamol )seule?",
    "seule?",
]

activities_of_daily_life = dict(
    source="other_activities",
    regex=[
        r"(?:actes|activites) de la vie quotidienne",
        "activites? quotidiennes?",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="altered_complements",
            regex=make_assign_regex(ALTERED_COMPLEMENTS),
            window=(-5, 3),
        ),
        dict(
            name="healthy_complements",
            regex="(autonome)",
            window=(-5, 3),
        ),
    ],
)

other_specific_qualifers = dict(
    source="other_qualifiable",
    regex=[
        r"moyens? de transports?",
        r"deplacements? tc",
        "budget",
        "gestion des comptes",
        "impots",
        r"transports? en commun",
        r"prise (?:des|du) repas",
        r"prise (?:des|du) traitements?",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="altered_complements",
            regex=make_assign_regex(ALTERED_COMPLEMENTS),
            window=(-5, 5),
        ),
        dict(
            name="severe_complements",
            regex=make_assign_regex(SEVERE_COMPLEMENTS),
            window=(-5, 5),
        ),
        dict(
            name="healthy_complements",
            regex=make_assign_regex(HEALTHY_COMPLEMENTS),
            window=(-5, 3),
        ),
    ],
)


administrative = dict(
    source="other_administrative",
    regex=[
        r"administrati(?:f|ve)",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="altered_complements",
            regex=make_assign_regex(ALTERED_COMPLEMENTS),
            window=(-4, 3),
        ),
        dict(
            name="severe_complements",
            regex=make_assign_regex(SEVERE_COMPLEMENTS),
            window=(-4, 3),
        ),
        dict(
            name="healthy_complements",
            regex=make_assign_regex(HEALTHY_COMPLEMENTS),
            window=(-4, 3),
        ),
    ],
    exclude=[
        dict(
            regex=["votre", "vos", "vous", r"\bdonnees", r"medico[\s-]?", "cadre"],
            window=(-5, 5),
        ),
        dict(regex=["hopital"], window=(-10, 10)),
    ],
)

AMBIGUOUS_ITEMS_CONTEXT = [
    "hygiene",
    "douche",
    "habillage",
    "toilettes?",
    "transfert",
    "locomotion",
    "continence",
    "repas",
    "manger",
    "mange",
    r"(?:gestion|prise) des medicaments",
    r"(?:gestion|prise) medicamenteuse",
    "traitement",
    r"preparation et distribution des medicaments",
    r"prepar(?:ation|er) (?:des|les) repas",
    "cuisine",
    "courses?",
    r"administrati(?:f|ve)",
    r"\bmenage",
    "etat fonctionnel",
    "autonomie",
    "conduite automobile",
    r"(?:actes|activites) de la vie quotidienne",
    "activites? quotidiennes?",
    r"moyens? de transports?",
    r"deplacements? tc",
    "budget",
    "gestion des comptes",
    "impots",
    r"transports? en commun",
    "plan fonctionnel",
]

ambiguous_items = dict(
    source="other_ambiguous_items",
    regex=[
        "repas",
        "manger",
        "mange",
        "toilette",
        "douche",
        "transfert",
        "traitement",
        r"medicaments?\b",
        "cuisine",
        "courses",
        r"\bmenage",
        "locomotion",
        "continence",
        "conduite",
        "hygiene",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="altered_complement",
            regex=make_assign_regex(ALTERED_COMPLEMENTS),
            window=(-4, 3),
        ),
        dict(
            name="severe_complement",
            regex=make_assign_regex(SEVERE_COMPLEMENTS),
            window=(-4, 3),
        ),
        dict(
            name="healthy_complement",
            regex=make_assign_regex(HEALTHY_COMPLEMENTS),
            window=(-4, 3),
        ),
    ],
    include=dict(
        regex=make_assign_regex(
            ALTERED_COMPLEMENTS
            + HEALTHY_COMPLEMENTS
            + SEVERE_COMPLEMENTS
            + AMBIGUOUS_ITEMS_CONTEXT
        ),
        window=(-4, 3),
    ),
)

readaptation = dict(
    source="other_readaptation",
    regex=["readaptation"],
    regex_attr="NORM",
    exclude=[
        dict(regex=["traitement", "vesicale?", "orthophonique"], window=5),
        dict(regex=["service", "unite", "medecine"], window=-10),
    ],
)

default_patterns = normalize_space_characters(
    [
        other_orth,
        severe,
        other,
        other_specific_qualifers,
        activities_of_daily_life,
        readaptation,
        house_stay,
        ambiguous_items,
        administrative,
        severe_orth,
        mild,
    ]
)
