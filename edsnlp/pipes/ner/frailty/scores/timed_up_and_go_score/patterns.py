from ..utils import float_regex

default_patterns = [
    dict(
        source="tug",
        regex=[r"\btg?ug\b", r"timed (get\s? )?up and go"],
        assign=[
            dict(
                name="value",
                regex=rf"({float_regex})",
                window=(0, 7),
                reduce_mode="keep_first",
            ),
        ],
    )
]
