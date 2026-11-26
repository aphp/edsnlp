from ...utils import normalize_space_characters
from ..utils import assigned_duration

default_patterns = normalize_space_characters(
    [
        dict(
            source="chair_stand",
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
            # We skip repetitions in one minute, e.g. `sur 1 minute : 17`, another test
            exclude=dict(
                regex=r"(?:sur|en)\s+1\s*(?:minute|mn|min)\b(?!\s*\d)|/\s*min",
                window=(0, 6),
            ),
        )
    ]
)
