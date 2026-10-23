from ...utils import normalize_space_characters
from ..utils import assigned_score_value

default_patterns = normalize_space_characters(
    [
        dict(
            source="sppb",
            regex=[r"\bsppb\b", "short physical performance battery"],
            assign=[
                dict(
                    name="value",
                    regex=assigned_score_value,
                    window=(0, 7),
                    reduce_mode="keep_first",
                ),
            ],
        )
    ]
)
