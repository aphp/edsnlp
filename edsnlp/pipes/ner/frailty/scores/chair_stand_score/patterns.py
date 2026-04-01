from ...utils import normalize_space_characters
from ..utils import float_regex

default_patterns = normalize_space_characters(
    [
        dict(
            source="sppb",
            regex=[r"(?:5 |cinq )?levers de chaise"],
            assign=[
                dict(
                    name="value",
                    regex=rf"({float_regex})",
                    window=(0, 7),
                    reduce_mode="keep_last",
                ),
            ],
        )
    ]
)
