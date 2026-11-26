from ..utils import assigned_score_value

default_patterns = [
    dict(
        source="g8",
        regex=r"\bg8\b|oncodage",
        assign=[
            dict(
                name="value",
                regex=assigned_score_value,
                window=(0, 7),
                replace_entity=False,
                reduce_mode="keep_first",
            ),
        ],
    )
]
