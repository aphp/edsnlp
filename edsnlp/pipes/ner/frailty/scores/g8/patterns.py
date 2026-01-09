from ..utils import float_regex, int_regex

default_patterns = [
    dict(
        source="g8",
        regex=r"\bg8\b|oncodage",
        assign=[
            dict(
                name="value",
                regex=rf"((?<![/g]){float_regex})(?:/{int_regex})?",
                window=(0, 7),
                replace_entity=False,
                reduce_mode="keep_first",
            ),
        ],
    )
]
