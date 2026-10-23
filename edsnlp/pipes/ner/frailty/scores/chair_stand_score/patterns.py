from ...utils import normalize_space_characters
from ..utils import assigned_duration

default_patterns = normalize_space_characters(
    [
        dict(
            source="sppb",
            regex=[r"(?:5 |cinq )?levers de chaise"],
            assign=[
                dict(
                    name="value",
                    # We skip the count of `5 levers de chaise`, which is not a duration
                    regex=rf"{assigned_duration}(?!\s*levers?\b)",
                    window=(0, 7),
                    reduce_mode="keep_last",
                ),
            ],
        )
    ]
)
