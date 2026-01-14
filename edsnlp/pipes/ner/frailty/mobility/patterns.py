from ..utils import make_assign_regex, make_status_assign, normalize_space_characters

healthy = dict(
    source="healthy",
    regex=[
        "autonomie motrice",
        "sort dans le jardin",
        "sort tous les jours",
        "sort regulierement",
        "se verticalise",
        "activite physique quotidienne",
    ],
    regex_attr="NORM",
)
severe = dict(
    source="severe",
    regex=[
        "lit medicalise",
        "grabataire",
        "alitement",
        "leve malade",
        r"\balitee?\b",
        r"pose de ridelles",
        "chaise roulante",
        "fauteuil roulant",
    ],
    regex_attr="NORM",
)
altered = dict(
    source="altered",
    regex=[
        r"troubles? de l'equilibre",
        r"syndromes? extra(?:[\s-])?pyramida(?:l|ux)",
        r"signes? extra(?:[\s-])?pyramida(?:l|ux)",
        r"syndromes? parkinsonn?iens?",
        "maladie de parkinson",
        "fonte musculaire",
        "sarcopenie",
    ],
    regex_attr="NORM",
)

sarcopenia = dict(
    source="altered_sarcopenia",
    regex=["sarcopenie"],
    regex_attr="NORM",
    assign=dict(
        name="severe",
        regex=make_assign_regex(["severe"]),
        window=2,
    ),
)

altered_equilibrium = dict(
    source="altered_equilibrium",
    regex=[
        "desequilibre",
    ],
    regex_attr="NORM",
    exclude=dict(
        regex=["profondeur", "diabete"],
        window=5,
    ),
)

reeducation = dict(
    source="other_reeducation",
    regex=["reeducation"],
    regex_attr="NORM",
    exclude=[
        dict(regex=["traitement", "vesicale?", "orthophonique"], window=5),
        dict(regex=["service", "unite", "medecine"], window=-10),
    ],
)

other = dict(
    source="other",
    regex=[
        "mobilite",
        r"appui (mono|uni)?podal",
        "locomotion",
        "test(?:ing)? moteur",
        "force motrice",
        "station debout",
        "station tandem",
        r"(?:sur le )plan (?:loco[\s-]?)?moteur",
        "travail sur l'equilibre",
        r"lutter contre l'enraidissement",
        "renforcement musculaire",
        "genee? dans son activite physique",
        "force de prehension",
    ],
    regex_attr="NORM",
    assign=make_status_assign(-4, 4),
)

mild = dict(
    source="mild",
    regex=[
        r"\bcanne\b",
        r"syndrome post[\s-]chute",
        "deambulateur",
        r"rollator",
        r"\bchutes?( a repetitions?)?",
        "petit perimetre",
        "se mobilisant difficilement",
    ],
    regex_attr="NORM",
)

ralentissement = dict(
    source="mild_ralentissement",
    regex=["ralentissement"],
    assign=dict(
        name="complement",
        regex=make_assign_regex(["moteur", "marche"]),
        window=5,
        required=True,
    ),
)

WALKING_ALTERED_COMPLEMENTS = [
    "peu",
    "a petits? pas",
    "troubles?",
    "renforcer",
    "pertes? des capacites?",
    "incapable",
    "precaire",
    "instable",
    "instabilite",
    "limite",
    "difficile",
    "(?<!sans )difficultes?",
    "(?<!sans )aides?",
]

WALKING_HEALTHY_COMPLEMENTS = [
    "sans aides?(?: techniques?)?",
    "sans difficultes?",
    "sans canne",
    r"\bcapable",
    "amelioree?s?",
    r"amelior(?:er|ation)",
    "normale?",
    "normalement",
    "correcte?s?",
    "bon(?:ne)?",
    "reprise",
    r"\bstable",
    "seule?",
]

walking = dict(
    source="walking",
    regex=[r"\bmarche", "deplacement", r"equilibre\b", "se deplace"],
    regex_attr="NORM",
    assign=[
        dict(
            name="altered_complement",
            regex=make_assign_regex(WALKING_ALTERED_COMPLEMENTS),
            window=(-4, 6),
        ),
        dict(
            name="healthy_complement",
            regex=make_assign_regex(WALKING_HEALTHY_COMPLEMENTS),
            window=(-4, 6),
        ),
    ],
    include=[
        dict(
            name="walking_complement",
            regex=make_assign_regex(
                WALKING_ALTERED_COMPLEMENTS + WALKING_HEALTHY_COMPLEMENTS
            ),
            window=(-4, 6),
        )
    ],
    exclude=dict(regex=["diabete", "profondeur"], window=3),
)

walking_perimeter = dict(
    source="other_perimeter",
    regex=["perimetre de marche"],
    regex_attr="NORM",
    assign=[
        dict(
            name="altered_complement",
            regex=make_assign_regex(
                ["reduit", "retreci", "limite", "inferieur a (?:5|cinq) ?m"]
            ),
        ),
        dict(name="healthy_complement", regex=make_assign_regex(["normal"])),
    ],
)

sustentation = dict(
    source="other_sustentation",
    regex=["polygone de sustentation"],
    regex_attr="NORM",
    assign=dict(name="altered_retreci", regex="(retreci)", window=3),
)

default_patterns = normalize_space_characters(
    [
        healthy,
        altered,
        severe,
        other,
        mild,
        sarcopenia,
        ralentissement,
        walking,
        reeducation,
        sustentation,
        walking_perimeter,
        altered_equilibrium,
    ]
)
