from ..utils import make_assign_regex, make_include_dict_from_list, make_status_assign

healthy = dict(
    source="healthy",
    regex=[
        "ingesta normal",
        "bonne prise alimentaire",
    ],
    regex_attr="NORM",
)
severe = dict(
    source="severe",
    regex=[
        "denutrition severe",
        r"cache(ctique|xie)",
    ],
    regex_attr="NORM",
)
altered = dict(
    source="altered",
    regex=[
        "anorexie",
        r"denutri(e?|tion)",
        r"difficultes? a s'alimenter",
        r"fausses? routes?",
        "ingesta anormal",
        "perte d'appetit",
        r"(?:a )?perdu \d+\s?kgs?",
        "perte ponderale",
        "reduction prise alimentaire",
        "sarcopenie",
        "amaigrissement",
        "prise en charge dietetique",
    ],
    regex_attr="NORM",
)

mild = dict(
    source="mild",
    regex=[
        "support nutritionnel",
        r"supplements? nutritionnels?",
        r"supplementation (?:alimentaire|nutritionnelle)",
    ],
    regex_attr="NORM",
)

consumption = dict(
    source="other_consumption",
    regex=["courses? alimentaires?", "aliments?", "nourriture"],
    regex_attr="NORM",
    assign=[
        dict(
            name="okay_complement",
            regex=make_assign_regex(["consomme", "mange"]),
            window=(-4, 4),
        ),
        dict(
            name="bad_complement",
            regex=make_assign_regex(
                [r"ne consomme (?:pas|plus)", r"ne mange (?:pas|plus)"]
            ),
            window=(-4, 4),
        ),
    ],
    include=dict(regex=make_assign_regex(["consomme", "mange"]), window=(-4, 4)),
)

altered_orth = dict(
    source="altered_orth",
    regex=[r"\bFR\b"],  # TODO : attention à "fréquence respiratoire"
    regex_attr="ORTH",
    exclude=dict(
        regex=r"\d+",
        window=2,
    ),
)

ca = dict(
    source="mild_ca",
    regex=[
        "cno",
        r"complements? (alimentaires?|nutritionnels?)(?: ora(l|ux))?",
        r"complements? ora(?:l|ux)",
    ],
    regex_attr="NORM",
    assign=dict(
        name="mild_ca",
        regex=make_assign_regex(
            ["ne prend pas", "ne prend plus", "non prise", "pas pris"]
        ),
        window=(-3, 3),
    ),
)

other = dict(
    source="other",
    regex=[
        "apports? caloriques?",
        "gene alimentaire",
        "renutrition",
        r"\brealimentation",
        r"s'alimente",
        r"reprise (alimentaire|de l'appetit)",
        "enrichissement de l'alimentation",
        "appetit",
        "amelioration nutritionnelle",
    ],
    regex_attr="NORM",
)

status = dict(
    source="other_status",
    regex=[
        "etat nutritionnel",
        "statut nutritionnel",
    ],
    regex_attr="NORM",
    assign=make_status_assign(altered_level="mild"),
)

weight = dict(
    source="other_weight",
    regex=[
        "poids",
    ],
    regex_attr="NORM",
    assign=make_status_assign()
    + [
        dict(
            name="altered_loss",
            regex=make_assign_regex(["perte"]),
            window=-3,
        ),
        dict(
            name="healthy_gain",
            regex=make_assign_regex(["prise"]),
            window=-3,
        ),
    ],
    include=make_include_dict_from_list(
        make_status_assign()
        + [
            dict(
                name="altered_loss",
                regex=make_assign_regex(["perte"]),
                window=-3,
            ),
            dict(
                name="healthy_gain",
                regex=make_assign_regex(["prise"]),
                window=-3,
            ),
        ],
    ),
)

kilograms = dict(
    source="other_kg",
    regex=["kgs?", r"kilos?(?:[\s-]?grammes?)?"],
    regex_attr="NORM",
    assign=[
        dict(
            name="mild_complements",
            regex=make_assign_regex(["perte"]),
            window=(-3, 3),
        ),
        dict(
            name="healthy_complement",
            regex=make_assign_regex(["prise"]),
            window=(-3, 3),
        ),
    ],
    include=dict(regex=make_assign_regex(["perte", "prise"]), window=(-3, 3)),
)

vitamin = dict(
    source="other_vitamin",
    regex=[
        r"(?:vit(?:amines?)? ?)?b9\b",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="altered_carence",
            regex=make_assign_regex(["carences?"]),
            window=-7,
            required=True,
        )
    ],
)

TROUBLE_COMPLEMENTS = ["deglutition", "nutritionnels?"]
troubles = dict(
    source="altered_troubles",
    regex=["troubles?"],
    regex_attr="NORM",
    assign=dict(
        name="trouble_complement",
        regex=make_assign_regex(TROUBLE_COMPLEMENTS),
        window=6,
        required=True,
    ),
)

ALIMENTATION_COMPLEMENTS = [
    "hyperprotidique",
    "hypercalorique",
    "hyperenergetique",
    "hp",
    "hc",
]
alimentation = dict(
    source="other_alimentation",
    regex=["alimentation", "regime"],
    regex_attr="NORM",
    assign=[
        dict(
            name="mild_complements",
            regex=make_assign_regex(ALIMENTATION_COMPLEMENTS),
            window=4,
        ),
        dict(name="per_os", regex=make_assign_regex(["per os"]), window=4),
    ],
    include=dict(
        regex=make_assign_regex(ALIMENTATION_COMPLEMENTS + ["per os"]), winodw=(-4, 4)
    ),
)

default_patterns = [
    healthy,
    altered,
    severe,
    other,
    alimentation,
    troubles,
    vitamin,
    ca,
    weight,
    kilograms,
    altered_orth,
    consumption,
    status,
    mild,
]
