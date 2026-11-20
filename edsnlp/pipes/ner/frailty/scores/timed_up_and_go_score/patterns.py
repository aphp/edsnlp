from ..utils import assigned_duration

default_patterns = [
    dict(
        source="tug",
        regex=[r"\btg?ug\b", r"timed (get\s? )?up and go"],
        assign=[
            dict(
                name="value",
                regex=assigned_duration,
                window=(0, 7),
                reduce_mode="keep_first",
            ),
        ],
    )
]
