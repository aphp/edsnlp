from ...utils import normalize_space_characters
from ..utils import float_regex

default_patterns = normalize_space_characters(
    [
        dict(
            source="walk_speed",
            regex=[
                "vitesse de marche",
            ],
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
)
