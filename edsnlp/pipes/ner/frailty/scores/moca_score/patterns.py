from ..utils import assigned_score_value, score_limit

moca_value = r"(?<!\bversion\s)" + assigned_score_value

default_patterns = [
    dict(
        source="moca",
        regex=r"\bmoca\b",
        assign=[
            dict(
                name="value",
                regex=moca_value,
                window=(0, 35),
                replace_entity=False,
                reduce_mode="keep_last",
            ),
            score_limit,
        ],
    )
]
