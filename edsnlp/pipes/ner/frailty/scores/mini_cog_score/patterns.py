from ..utils import assigned_score_value

default_patterns = [
    dict(
        source="mini_cog",
        regex=r"\bmini[\s-]+cog\b",
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
