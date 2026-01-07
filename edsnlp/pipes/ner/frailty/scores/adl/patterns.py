from ..utils import float_regex, int_regex

default_patterns = [
    dict(
        source="adl",
        regex=r"\badl\b",
        assign=[
            dict(
                name="value",
                regex=rf"((?<!/){float_regex})(?:/{int_regex})?",
                window=(0, 35),
                replace_entity=False,
                reduce_mode="keep_last",
            ),
            dict(name="limit_iadl", regex="(iadl)", window=(0, 35)),
        ],
    )
]
