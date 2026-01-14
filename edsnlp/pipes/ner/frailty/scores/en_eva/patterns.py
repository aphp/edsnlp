from ...utils import normalize_space_characters
from ..utils import float_regex, int_regex

default_patterns = normalize_space_characters(
    [
        dict(
            source="en_eva",
            regex=[r"\bEN\b", r"\bEVA\b"],
            assign=[
                dict(
                    name="value",
                    regex=rf"((?<!/){float_regex})(?:/{int_regex})?",
                    window=(0, 7),
                    reduce_mode="keep_first",
                ),
            ],
        )
    ]
)
