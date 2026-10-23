from ..utils import assigned_score_value, score_limit

default_patterns = [
    dict(
        source="bref",
        regex=r"\bbref\b",
        assign=[
            dict(
                name="value",
                regex=assigned_score_value,
                window=(0, 7),
                replace_entity=False,
                reduce_mode="keep_first",
            ),
            score_limit,
        ],
    )
]
