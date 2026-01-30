from ..utils import make_assign_regex, make_status_assign, normalize_space_characters

healthy = dict(
    source="healthy",
    regex=[
        "aidante? principale?",
        "soutien familiale?",
        "bien entouree?",
    ],
    regex_attr="NORM",
)
healthy_orth = dict(
    source="healthy_orth",
    regex=[
        r"\btravaille\b",
    ],
    regex_attr="ORTH",
)
severe = dict(
    source="severe",
    regex=[
        r"soins? de nursing",
        r"tut(elles?|eur|rice)",
        r"curat(?:elle|eur|rice)",
        "mandataire professionnel",
        "usld",
        "sauvegarde de justice",
        "maison de retraite",
        "protection juridique",
        r"intitution(nalisee?)?",
        r"\bmadd\b",
    ],
    regex_attr="NORM",
)
altered = dict(
    source="altered",
    regex=[
        r"\bapa\b",
        "peu entouree?",
    ],
    regex_attr="NORM",
)

mild = dict(
    source="mild",
    regex=[
        r"residence (autonomie|senior)",
        "foyer logement",
    ],
)

severe_potential_fp = dict(
    source="severe_potential_fp",
    regex=[
        "ehpad",
    ],
    regex_attr="NORM",
    exclude=dict(regex=["equipe"], window=(-2, 2)),
)
severe_orth = dict(
    source="severe_orth",
    regex=[
        r"\bEPHAD\b",
    ],
    regex_attr="ORTH",
)

house_stay = dict(
    source="other_stay",
    regex=["maintien au? domicile", r"\bmad\b"],
    regex_attr="NORM",
    assign=dict(
        name="severe",
        regex=make_assign_regex(["difficile", "difficulte"]),
        window=(-2, 2),
    ),
)

isolation = dict(
    source="altered_isolation",
    regex=[
        r"isole(e?|ment)",
    ],
    exclude=[
        dict(regex=["progression"], window=-5),
        dict(regex=[r"\besa\b", "nodule", "hepatomegalie"], window=-2),
        dict(regex=["contact", "bmr"], window=4),
    ],
    regex_attr="NORM",
)
other = dict(
    source="other",
    regex=[
        "habitation familiale",
        r"aide[\s-]menagere",
        r"aides? au? domicile",
        r"aux(iliaires?)? de vie",
        r"\badv\b",
        "bilan social",
        "(?:sur le )?plan social",
        r"vit seule?",
        r"vivait seule?",
        "activite professionnelle",
    ],
    regex_attr="NORM",
)

HEALTHY_FAMILY_COMPLEMENTS = ["vit avec", "presente?s?"]
ALTERED_FAMILY_COMPLEMENTS = [
    "decedee?s?",
    "morte?s?",
    "deces",
    r"ne(?: se)? voit (?:plus|pas)",
    r"ne(?: se)? parle (?:plus|pas)",
    "conflit",
]

family_members = dict(
    source="other_family",
    regex=[
        r"\bmari\b",
        "femme",
        r"epou(?:x|se)",
        r"\bcompagn(?:e|on)",
        "concubine?",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="healthy_family",
            regex=make_assign_regex(HEALTHY_FAMILY_COMPLEMENTS),
            window=(-3, 4),
        ),
        dict(
            name="altered_family",
            regex=make_assign_regex(ALTERED_FAMILY_COMPLEMENTS),
            window=(-3, 4),
        ),
    ],
    include=dict(
        regex=make_assign_regex(
            HEALTHY_FAMILY_COMPLEMENTS + ALTERED_FAMILY_COMPLEMENTS
        ),
        window=(-3, 4),
    ),
    exclude=dict(
        regex=["resorbables?", "ablation", r"chirurgi(?:e|cal|caux)"], window=(-4, 4)
    ),
)
children = dict(
    source="healthy_children",
    regex=[
        r"(\d+ ?)?(?:petits?[\s-]?)?fils",
        r"(\d+ ?)?garcons?",
        r"(\d+ ?)?(?:petites?[\s-]?)?filles?",
        r"(\d+ ?)?(?:petits?[\s-]?)?enfants?",
    ],
    assign=dict(
        name="healthy_children",
        regex=make_assign_regex(HEALTHY_FAMILY_COMPLEMENTS),
        window=(-3, 4),
        required=True,
    ),
)

ambiguous_family = dict(
    source="other_ambiguous_family",
    regex=[
        "famille",
        r"(\d+ ?)?\bfreres?",
        r"(\d+ ?)?\bsoeurs?",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="healthy_family",
            regex=make_assign_regex(HEALTHY_FAMILY_COMPLEMENTS),
            window=(-3, 4),
        ),
        dict(
            name="altered_family",
            regex=make_assign_regex(ALTERED_FAMILY_COMPLEMENTS),
            window=(-3, 4),
        ),
    ],
    exclude=dict(
        regex=["resorbables?", "ablation", r"chirurgi(?:e|cal|caux)"], window=(-4, 4)
    ),
    include=dict(
        regex=make_assign_regex(
            HEALTHY_FAMILY_COMPLEMENTS + ALTERED_FAMILY_COMPLEMENTS
        ),
        window=(-3, 4),
    ),
)

status = dict(
    source="other_status",
    regex=["etat social", "statut social"],
    regex_attr="NORM",
    assign=make_status_assign(),
)

default_patterns = normalize_space_characters(
    [
        healthy,
        healthy_orth,
        altered,
        severe_orth,
        isolation,
        severe,
        mild,
        other,
        family_members,
        children,
        ambiguous_family,
        house_stay,
        severe_potential_fp,
        status,
    ]
)
