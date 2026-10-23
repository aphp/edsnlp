from ...utils import make_assign_regex
from ..utils import assigned_score_value

default_patterns = [
    dict(
        source="ps",
        regex=[r"\b(?:ecog[\s-]?)?ps\b", r"\boms\b", r"ecog"],
        assign=[
            dict(
                name="value",
                regex=assigned_score_value,
                window=(0, 7),
                replace_entity=False,
                reduce_mode="keep_first",
            ),
        ],
        exclude=dict(regex=make_assign_regex([r"\btel\b", "telephone"])),
    )
]
