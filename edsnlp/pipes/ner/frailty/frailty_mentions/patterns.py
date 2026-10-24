from ..utils import (
    ALTERED_STATUS_COMPLEMENTS,
    HEALTHY_STATUS_COMPLEMENTS,
    make_assign_regex,
    normalize_space_characters,
)

# Since the frailty of a function, a tissue or a gene is not the patient frailty, we
# skip it, e.g. `peau fragile`, `fragilite osseuse` or `syndrome de l'X fragile`
non_frailty = [
    dict(
        regex=make_assign_regex(
            [
                "thymique",
                "cogniti(?:f|ve)",
                "nutritionnel(?:le)?",
                "osseu",
                "cutane",
                "peau",
                "muqueu",
                "mnes",
                "memoire",
                "executi",
            ]
        )
    ),
    dict(regex=r"\bx\b", window=-2),
]

frail = dict(
    source="altered",
    regex=[r"\b(?:pre[\s-]?)?fragiles?\b"],
    regex_attr="NORM",
    exclude=non_frailty,
)

severe = dict(
    source="severe",
    regex=[
        r"(?:extremement|trop) fragiles?\b",
    ],
    regex_attr="NORM",
    exclude=non_frailty,
)

frailty = dict(
    source="other_frailty",
    regex=[r"\bfragilites?\b"],
    regex_attr="NORM",
    assign=[
        dict(
            name="healthy_complements",
            regex=make_assign_regex(HEALTHY_STATUS_COMPLEMENTS),
            window=(-3, 3),
        ),
        dict(
            name="altered_complements",
            regex=make_assign_regex(ALTERED_STATUS_COMPLEMENTS),
            window=(-3, 3),
        ),
    ],
    exclude=non_frailty,
)


default_patterns = normalize_space_characters([frail, frailty, severe])
