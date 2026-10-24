from ..utils import make_linked_score_value, score_limit

default_patterns = [
    dict(
        source="bref",
        regex=r"\bbref\b",
        assign=[
            dict(
                name="value",
                regex=make_linked_score_value(r"bref"),
                window=(0, 7),
                replace_entity=False,
                reduce_mode="keep_first",
            ),
            score_limit,
        ],
    )
]
