from ..utils import (
    ALTERED_STATUS_COMPLEMENTS,
    HEALTHY_STATUS_COMPLEMENTS,
    make_assign_regex,
    normalize_space_characters,
)

frail = dict(
    source="altered",
    regex=["fragile"],
    regex_attr="NORM",
    exclude=dict(
        regex=make_assign_regex(["thymique", "cogniti(?:f|ve)", "nutritionnel(?:le)?"])
    ),
)

severe = dict(
    source="severe",
    regex=[
        r"(?:extremement|trop) fragile",
    ],
    regex_attr="NORM",
    exclude=dict(
        regex=make_assign_regex(["thymique", "cogniti(?:f|ve)", "nutritionnel(?:le)?"])
    ),
)

frailty = dict(
    source="other_frailty",
    regex=["fragilite"],
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
    exclude=dict(
        regex=make_assign_regex(["thymique", "cogniti(?:f|ve)", "nutritionnel(?:le)?"])
    ),
)


default_patterns = normalize_space_characters([frail, frailty, severe])
