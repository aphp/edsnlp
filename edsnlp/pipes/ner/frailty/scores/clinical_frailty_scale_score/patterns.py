from ..utils import assigned_score_value

default_patterns = [
    dict(
        source="ps",
        regex=[r"\brockwood\b", "clinical frailty scale"],
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
