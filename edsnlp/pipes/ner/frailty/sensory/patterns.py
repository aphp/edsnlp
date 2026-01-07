from ..utils import make_status_assign, normalize_space_characters

healthy = dict(
    source="healthy",
    regex=[r"acuite (?:visuelle|auditive) normale"],
    regex_attr="NORM",
)
severe = dict(
    source="severe",
    regex=[
        r"\bcecite\b",
        "aveugle",
        "surdite",
    ],
    regex_attr="NORM",
)
altered = dict(
    source="altered",
    regex=[
        r"appareillages? auditifs?",
        "lunettes",
        r"baisse d'acuite visuelle",
    ],
    regex_attr="NORM",
)
mild = dict(
    source="mild",
    regex=[
        "hypoacousie",
        "presbyacousie",
        "difficultes? visuelles?",
    ],
    regex_attr="NORM",
)
assourdi = dict(
    source="severe_assourdi",
    regex=[
        r"sourde?",
    ],
    exclude=dict(regex=[r"coeurs?", r"cardiaques?"], window=-3),
    regex_attr="NORM",
)
bav = dict(
    source="altered_bav",
    regex=[r"\bBAV\b"],
    exclude=dict(
        regex=[
            "pacemaker",
            "coeur",
            r"cardiaques",
            "complet",
            "nephrologue",
            "cardiologue",
            "degre",
        ],
        window=(-15, 10),
        regex_attr="NORM",
    ),
    regex_attr="ORTH",
)
other = dict(
    source="other",
    regex=[r"bilan ophtalmo(?:logique)?", "avis ophtalmologique"],
    regex_attr="NORM",
)

status = dict(
    source="other_status",
    regex=[
        "statut visuel",
        "etat visuel",
        "statut auditif",
        "etat auditif",
        "statut sensoriel",
        "etat sensoriel",
    ],
    regex_attr="NORM",
    assign=make_status_assign(),
)

default_patterns = normalize_space_characters(
    [
        other,
        healthy,
        altered,
        mild,
        severe,
        assourdi,
        bav,
        status,
    ]
)
