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
        "etat thymique correct",
        "a le moral",
        "euthymique",
    ],
    regex_attr="NORM",
)
severe = dict(
    source="severe",
    regex=[
        "perte de l'elan vital",
        r"idees? suicidaires?",
        r"idees? noires?",
    ],
    regex_attr="NORM",
)
altered = dict(
    source="altered",
    regex=[
        r"angoissee?",
        "anhedonie",
        r"anti[\s-]?depresseur",
        r"anxieu(se|x)",
        "anxiete",
        "axiolytique",
        "depreciation",
        "depression",
        r"deprimee?",
        r"(?:anxio[-\s]?)?depressi(?:f|ve)",
        "souffrance psychique",
        r"syndrome (?:anxio[-\s]?)?depressif",
        "thymie basse",
        "tristesse",
        "triste",
        r"moral (?:moyen|bas)",
        r"anxiolytique",
        r"anxiolyse",
        r"dysthymi(?:e|que)",
        "desesperee?",
        "desespoir",
    ],
    regex_attr="NORM",
)


HEALTHY_SLEEP_COMPLEMENTS = HEALTHY_STATUS_COMPLEMENTS + ["nocturne", "nuit"]
ALTERED_SLEEP_COMPLEMENTS = ALTERED_STATUS_COMPLEMENTS + ["diurne", "jour"]
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
            name="sleep_good",
            regex=make_assign_regex(HEALTHY_SLEEP_COMPLEMENTS),
            window=(-4, 4),
        ),
        dict(
            name="sleep_bad",
            regex=make_assign_regex(ALTERED_SLEEP_COMPLEMENTS),
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

troubles = dict(
    source="altered_troubles",
    regex=[r"(?<!bilan de )(?<!bilan )troubles?", "anomalies?"],
    regex_attr="NORM",
    assign=dict(
        name="trouble_complement",
        regex=make_assign_regex(["sommeil"]),
        window=6,
        required=True,
    ),
)

ralentissement = dict(
    source="other_ralentissement",
    regex=["ralentissement"],
    exclude=dict(regex=["transit"], window=(-2, 2)),
    assign=dict(name="complement", regex=make_assign_regex(["ideatoire"]), window=3),
)

morale = dict(
    source="other_morale",
    regex=[r"\bmoral\b"],
    regex_attr="NORM",
    assign=make_status_assign(-4, 4),
    include=make_include_dict_from_list(make_status_assign(-4, 4)),
)

other = dict(
    source="other",
    regex=[
        "avis psychiatrique",
        "antecedent psy",
        "thymie",
        "(?:sur le )?plan thymique",
        "examen psychiatrique",
        "persecution",
        r"nuits? difficiles?",
        "delire",
        "delirant",
        "insomnie",
        "somnolence",
        "psychiatre",
        "psychologue",
        "tendance a l'endormissement",
        r"\bapathi(?:e|que)",
        r"hallucinations?",
        "clinophilie",
    ],
    regex_attr="NORM",
)

status = dict(
    source="other_status",
    regex=[
        "statut thymique",
        "etat thymique",
        r"etat (?:neuro[\s-]?)?psychologique",
        r"statut (?:neuro[\s-]?)?psychologique",
    ],
    assign=make_status_assign(),
)

default_patterns = normalize_space_characters(
    [
        healthy,
        altered,
        severe,
        other,
        ralentissement,
        sleep,
        troubles,
        morale,
        night,
        status,
    ]
)
