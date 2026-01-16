from ..utils import float_regex, int_regex

default_patterns = [
    dict(
        source="mini_gds",
        regex=r"\bmini[\s-]+gds\b",
        assign=[
            dict(
                name="value",
                regex=rf"((?<!/){float_regex})(?:/{int_regex})?",
                window=(0, 35),
                replace_entity=False,
                reduce_mode="keep_last",
            ),
            dict(name="limit_iadl", regex=r"(\biadl\b)", window=(1, 35)),
            dict(name="limit_adl", regex=r"(\badl\b)", window=(1, 35)),
            dict(name="limit_moca", regex=r"(\bmoca\b)", window=(1, 35)),
            dict(name="limit_mms", regex=r"(\bmmse?\b)", window=(1, 35)),
            dict(name="limit_bref", regex=r"(\bbref\b)", window=(1, 35)),
            dict(name="limit_gds", regex=r"(\bgds\b)", window=(1, 35)),
        ],
        exclude=dict(
            regex=["arteriel", "artere", r"\bph\b", r"\bps\b", "sang", "gaz"],
            window=(-4, 4),
        ),
    )
]
