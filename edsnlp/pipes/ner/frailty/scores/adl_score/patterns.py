from ..utils import assigned_score_value, score_limit

default_patterns = [
    dict(
        source="adl",
        regex=r"\badl\b",
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
    )
]
