from ..utils import (
    ALTERED_STATUS_COMPLEMENTS,
    HEALTHY_STATUS_COMPLEMENTS,
    make_assign_regex,
    make_include_dict_from_list,
    make_status_assign,
    normalize_space_characters,
)

healthy = dict(
    source="healthy",
    regex=[
        "langage fluent",
        r"ne s'est jamais perdue?",
        r"discours (?:clair|adapte)(?: et (?:adapte|clair))?",
        "discours spontane",
    ],
    regex_attr="NORM",
)
severe = dict(
    source="severe",
    regex=[
        "demence",
        "tncm",
        "alzheimer",
        r"\bdts complete",
        r"neuro[\s-]?degenerati(?:f|ve)",
        r"neuro[\s-]?denerati(?:f|ve)",  # Faute de frappe
        "syndrome dementiel",
    ],
    regex_attr="NORM",
)
altered = dict(
    source="altered",
    regex=[
        "agitation",
        "agressivite",
        r"aphasi(?:qu)?e",
        r"apraxi(?:qu)?e",
        "confusion",
        r"patiente? confuse?",
        r"\bapathi(?:e|que)",
        "deficit de la memoire",
        "atteinte de la memoire",
        "delirium",
        r"idees? delirantes?",
        r"hallucin(e|ations?)",
        r"syndrome (amnesique|confusionnel)",
        "tncm",
        "plainte mnesique",
        "plainte cognitive",
        r"syndromes? extra(?:[\s-])?pyramida(?:l|ux)",
        r"signes? extra(?:[\s-])?pyramida(?:l|ux)",
        "roue dentee",
        r"syndromes? parkinsonn?iens?",
        r"marche a petits? pas",
        "manque du mot",
        r"(?:syndrome )?dysexecuti(?:f|ve)",
        r"\boubli[es]?\b",
        r"pertes? de memoire",
        "se plaint de la memoire",
        "dyscalculie",
        r"n'arrive (?:pas|plus) a s'\sorienter",
        r"neuro(?:[\s-]?psycho)?logue",
        "anosognosie",
        "alteration cognitive",
        "accueil de jour",
    ],
    regex_attr="NORM",
    exclude=dict(regex="aigue?s?", window=(-3, 3)),
)

recognition = dict(
    source="altered_recognition",
    regex=["reconnaissance"],
    regex_attr="NORM",
    assign=dict(
        name="complement",
        regex=make_assign_regex(["perturbee?", "perturbation"]),
        window=(-4, 4),
        required=True,
    ),
)

consultation = dict(
    source="other_consultation",
    regex=[
        "consultation",
        r"\bcs\b",
    ],
    regex_attr="NORM",
    assign=dict(name="memoire", regex="(memoire)", window=3, required=True),
)

memory = dict(
    source="other_memory",
    regex=["memoire"],
    regex_attr="NORM",
    assign=make_status_assign(),
    include=make_include_dict_from_list(make_status_assign()),
)

TROUBLE_COMPLEMENTS_ALTERED = [
    "attentionnels?",
    "attention",
    "gnosiques?",
    "praxiques?",
    "praxies?",
    "memoire",
    "comprehension",
    "mnesiques?",
    "phasiques?",
    "fonctions? superieure?s?",
    "fonctions? executives?",
    "denomination",
    "flexibilite mentale",
    "sommeil paradoxal",
]
TROUBLE_COMPLEMENTS_SEVERE = [
    "jugement",
    "comportement",
]
troubles = dict(
    source="altered_troubles",
    regex=[
        r"(?<!bilan de )(?<!bilan )troubles?",
        "anomalies?",
        "difficultes?",
        "fragilites?",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="trouble_complement",
            regex=make_assign_regex(TROUBLE_COMPLEMENTS_ALTERED),
            window=6,
        ),
        dict(
            name="severe_trouble_complement",
            regex=make_assign_regex(TROUBLE_COMPLEMENTS_SEVERE),
            window=6,
        ),
    ],
    include=dict(
        regex=make_assign_regex(
            TROUBLE_COMPLEMENTS_ALTERED + TROUBLE_COMPLEMENTS_SEVERE
        ),
        window=6,
    ),
)

tnc = dict(
    source="altered_tnc",
    regex=[r"troubles? (?:neuro[\s-]?)?cogniti(?:f|ve)s?"],
    assign=[
        dict(name="mild", regex=r"(legers?|mineurs?)", window=2),
        dict(name="severe", regex="(majeurs?)", window=2),
    ],
)

ORIENTATION_COMPLEMENTS = [
    "temps",
    "espace",
    "temporel(?:le)?",
    "spatiale?",
    r"spatio[\s-]?temporelle",
]

desorientation = dict(
    source="altered_desorientation",
    regex=[
        r"(?:patiente? )?desorient(?:ee?|ation)",
    ],
    assign=[
        dict(
            name="complement",
            regex=make_assign_regex(ORIENTATION_COMPLEMENTS),
            window=15,
            reduce_mode=None,
        ),
        dict(
            name="severe_complement",
            regex=r"(\bcomplete(?:ment)?\b)",
            window=(-2, 15),
        ),
    ],
    regex_attr="NORM",
)


orientation_healthy = dict(
    source="healthy_orientation",
    regex=[
        r"(?:patiente? )?orientee?\b",
    ],
    assign=dict(
        name="complement",
        regex=make_assign_regex(ORIENTATION_COMPLEMENTS),
        window=15,
        reduce_mode=None,
    ),
    regex_attr="NORM",
)

orientation_other = dict(
    source="other_orientation",
    regex=[
        r"orientation",
    ],
    assign=make_status_assign(-2, 2)
    + [
        dict(
            name="complement",
            regex=make_assign_regex(ORIENTATION_COMPLEMENTS),
            window=15,
            reduce_mode=None,
            required=True,
        ),
    ],
    regex_attr="NORM",
)

cognitive_status = dict(
    source="other_cognitive_status",
    regex=[
        "etat cognitif",
        "statut cognitif",
        r"etat neuro(?:[\s-]?psycho)?logique",
        r"statut neuro(?:[\s-]?psycho)?logique",
    ],
    regex_attr="NORM",
    assign=make_status_assign(),
)

other = dict(
    source="other",
    regex=[
        r"evaluation neuro[\s-]?psychique",
        "tests? de memoire",
        "epreuves? de praxie",
        r"neuro[\s-]?cognitif",
        "fonctions? cognitives?",
        "(?:sur le )?plan cognitif",
        "(?:sur le )?plan attentionnel",
        "(?:sur le )?plan du comportement",
        "(?:sur le )?plan neurologique",
        r"(?:sur le )plan neuro[\s-]?psychologique",
        "praxies? gestuelles?",
        "praxies? constructives?",
        "profil cognitif",
        "evolution cognitive",
        "evolution comportementale",
        "tableau cognitif",
        "evaluation cognitive",
        r"psycho[-\s]?motricite",
        "remediation cognitive",
        r"test de l'horloge (?:(?!sans)\w+\s){0,5}succes",
        r"rappel des (?:trois|3) mots",
        "test des (?:cinq|5) mots",
        r"\bdubois \d{1,2}(?:/10)?",
        "suivi memoire",
        "orthophonie",
        r"contention (chimique|physique)",
        "stimulations? cognitives?",
        "nuits? difficiles?",
    ],
    regex_attr="NORM",
)


BILAN_COMPLEMENTS = [
    "memoire",
    r"troubles? (?:neuro[\s-]?)?cognitifs?",
    r"neuro[\s-]?psychologique",
    "cognitif",
]
bilan = dict(
    source="other_bilan",
    regex=["bilan"],
    regex_attr="NORM",
    assign=[
        dict(
            name="complement",
            regex=make_assign_regex(BILAN_COMPLEMENTS),
            required=True,
        )
    ],
)

ralentissement = dict(
    source="altered_ralentissement",
    regex=["ralentissement"],
    assign=dict(
        name="complement",
        regex=make_assign_regex(["ideatoire"]),
        window=3,
        required=True,
    ),
)


sleep = dict(
    source="other_sleep",
    regex=["sommeil", "endormissement", r"\bdort\b"],
    regex_attr="NORM",
    exclude=[
        dict(name="apnee", regex=["apnee"], window=-4),
        dict(
            name="medecine",
            regex=["gelule", "cachet", "comprime", "souhaite"],
            window=(-6, 6),
        ),
    ],
    assign=[
        dict(
            name="sleep_healthy",
            regex=make_assign_regex(HEALTHY_STATUS_COMPLEMENTS),
            window=(-4, 4),
        ),
        dict(
            name="sleep_bad",
            regex=make_assign_regex(ALTERED_STATUS_COMPLEMENTS),
            window=(-4, 4),
        ),
    ],
)

night = dict(
    source="other_night",
    regex=["nuits?"],
    regex_attr="NORM",
    assign=make_status_assign(-4, 4, priority=False),
    include=make_include_dict_from_list(make_status_assign(-4, 4)),
)

default_patterns = normalize_space_characters(
    [
        healthy,
        altered,
        severe,
        other,
        desorientation,
        orientation_healthy,
        orientation_other,
        ralentissement,
        cognitive_status,
        troubles,
        bilan,
        consultation,
        memory,
        recognition,
        sleep,
        night,
        tnc,
    ]
)
