from ...utils import make_assign_regex
from ..utils import make_linked_score_value

default_patterns = [
    dict(
        source="ps",
        regex=[r"\b(?:ecog[\s-]?)?ps\b", r"\b(?:oms|ecog)(?=\b|\d)"],
        assign=[
            dict(
                name="value",
                regex=[
                    make_linked_score_value(r"ecog|ps|oms"),
                    r"(?<=\b(?:ecog|oms))([0-4])(?!\w)",
                ],
                window=(0, 7),
                replace_entity=False,
                reduce_mode="keep_first",
            ),
        ],
        exclude=dict(regex=make_assign_regex([r"\btel\b", "telephone"])),
    )
]
