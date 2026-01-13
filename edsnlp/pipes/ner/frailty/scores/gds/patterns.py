from ..utils import float_regex, int_regex

default_patterns = [
    dict(
        source="gds",
        regex=r"(?<!-)\bgds\b",
        assign=[
            dict(
                name="value",
                regex=rf"((?<!/){float_regex})(?:/{int_regex})?",
                window=(0, 35),
                replace_entity=False,
                reduce_mode="keep_last",
            ),
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
