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
    # We skip the blind spot and blinded studies, e.g. `en double aveugle`
    exclude=dict(regex=[r"(?:tache|en|double)\s+aveugle"], window="words[-3:0]"),
)
altered = dict(
    source="altered",
    regex=[
        r"appareillages? auditifs?",
        r"baisse d'acuite visuelle",
    ],
    regex_attr="NORM",
)
glasses = dict(
    source="altered_glasses",
    regex=["lunettes"],
    regex_attr="NORM",
    # We skip nasal cannulas for oxygen therapy, e.g. `O2 2 L/min aux lunettes`
    exclude=dict(
        regex=[
            r"\b[0o]2\b",
            "oxyg",
            r"\bl\s*/\s*min",
            r"\bvni\b",
            "nasales?",
            "optiflow",
        ],
        window=(-10, 10),
    ),
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
        # The word boundaries keep surnames such as `Lesourd` out
        r"\bsourde?s?\b",
    ],
    exclude=dict(regex=[r"coeurs?", r"cardiaques?", r"douleurs?", "toux"], window=-3),
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
            # We skip atrioventricular blocks in ECG reports, e.g. `BAV 1`
            r"\becg\b",
            r"\bqrs\b",
            "sinusal",
            "holter",
            "bradycard",
            "conduction",
            r"\bbav (?:[123]|i{1,3})\b",
        ],
        window=(-15, 10),
        regex_attr="NORM",
    ),
    regex_attr="ORTH",
)
other = dict(
    source="other",
    regex=[
        r"bilan ophtalmo(?:logique)?",
        "avis ophtalmologique",
        "examen ophtalmologique",
    ],
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
        glasses,
        mild,
        severe,
        assourdi,
        bav,
        status,
    ]
)
