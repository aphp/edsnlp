from ..utils import assigned_score_value, score_limit

default_patterns = [
    dict(
        source="gds",
        regex=r"(?<!-)\bgds\b",
        assign=[
            dict(
                name="value",
                regex=assigned_score_value,
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
                r"\bps\b",
                "sang",
                "gaz",
                r"mini[\s-]*",
            ],
            window=(-4, 4),
        ),
    )
]
