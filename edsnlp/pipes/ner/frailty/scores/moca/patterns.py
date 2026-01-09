from ..utils import float_regex, int_regex

default_patterns = [
    dict(
        source="moca",
        regex=r"\bmoca\b",
        assign=[
            dict(
                name="value",
                regex=rf"((?<!/){float_regex})(?:/{int_regex})?",
                window=(0, 35),
                replace_entity=False,
                reduce_mode="keep_last",
            ),
            dict(name="limit_mms", regex=r"(\bmmse?\b)", window=(0, 35)),
        ],
    )
]
