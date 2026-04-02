from ..utils import float_regex, int_regex

default_patterns = [
    dict(
        source="bref",
        regex=r"\bbref\b",
        assign=[
            dict(
                name="value",
                regex=rf"((?<!/){float_regex})(?:/{int_regex})?",
                window=(0, 7),
                replace_entity=False,
                reduce_mode="keep_first",
            ),
            dict(name="limit_iadl", regex=r"(\biadl\b)", window=(1, 35)),
            dict(name="limit_adl", regex=r"(\badl\b)", window=(1, 35)),
            dict(name="limit_moca", regex=r"(\bmoca\b)", window=(1, 35)),
            dict(name="limit_mms", regex=r"(\bmmse?\b)", window=(1, 35)),
            dict(name="limit_bref", regex=r"(\bbref\b)", window=(1, 35)),
            dict(name="limit_gds", regex=r"(\bgds\b)", window=(1, 35)),
        ],
    )
]
