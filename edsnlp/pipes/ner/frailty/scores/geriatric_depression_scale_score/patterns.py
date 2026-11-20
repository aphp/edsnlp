from ..utils import make_linked_score_value, score_limit

default_patterns = [
    dict(
        source="gds",
        regex=r"(?<!-)(?<!mini[\s-])\bgds\b",
        assign=[
            dict(
                name="value",
                # Since GDS scores are whole numbers, we skip decimals, e.g. `GDS 7.39`
                regex=r"(?!\d+[.,]\d)" + make_linked_score_value(r"gds"),
                window=(0, 35),
                replace_entity=False,
                reduce_mode="keep_last",
            ),
            score_limit,
        ],
        exclude=dict(
            regex=[
                "arteriel",
                "artere",
                r"\bph\b",
                "sang",
                "gaz",
            ],
            window=(-4, 4),
        ),
    )
]
