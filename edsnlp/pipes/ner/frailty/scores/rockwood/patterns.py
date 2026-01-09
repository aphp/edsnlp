from ..utils import float_regex, int_regex

default_patterns = [
    dict(
        source="ps",
        regex=r"\brockwood\b",
        assign=[
            dict(
                name="value",
                regex=rf"((?<!/){float_regex})(?:/{int_regex})?",
                window=(0, 7),
                replace_entity=False,
                reduce_mode="keep_first",
            ),
        ],
    )
]
