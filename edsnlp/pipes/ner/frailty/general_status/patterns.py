from ..utils import HEALTHY_STATUS_COMPLEMENTS, make_assign_regex, make_status_assign

severe = dict(
    source="severe",
    regex=[
        r"(?:extremement|trop) fragile",
        "tres asthenique",
        r"agoni(?:e|que)",
        r"\baeg\b",
        r"prise en charge de confort",
    ],
    regex_attr="NORM",
)
altered = dict(
    source="altered",
    regex=[
        r"astheni(?:e|que)",
        r"fatiguee?",
        "fatigu?abilite",
        "plus la force de se lever",
        "l'etat du patient se degrade",
    ],
    regex_attr="NORM",
)


other = dict(
    source="other",
    regex=[
        "etat general",
        r"amelioration de l'etat general",
    ],
    regex_attr="NORM",
    assign=make_status_assign(),
)

general_status = dict(
    source="other",
    regex=[
        "etat general",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="severe_status",
            regex=make_assign_regex(
                ["alteration", "alteree?", "degradation", "degradee?"]
            ),
            window=(-3, 3),
        ),
        dict(
            name="healthy_status",
            regex=make_assign_regex(HEALTHY_STATUS_COMPLEMENTS),
            window=(-3, 3),
        ),
        dict(
            name="altered_status",
            regex=make_assign_regex(["anormale?", "troubles?"]),
            window=(-3, 3),
        ),
    ],
)

default_patterns = [altered, severe, other]
