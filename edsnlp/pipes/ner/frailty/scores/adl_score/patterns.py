from ..utils import assigned_score_value, score_limit

# We skip item values, e.g. `habillage 1`, since they are not the Katz total
adl_value = (
    r"(?<!\b(?:hygiene|habillage|continence|transferts?|toilettes?|wc|repas|"
    r"locomotion|alimentation)\s*[:=]?\s*)" + assigned_score_value
)

default_patterns = [
    dict(
        source="adl",
        regex=r"\badl\b",
        assign=[
            dict(
                name="value",
                regex=adl_value,
                window=(0, 35),
                replace_entity=False,
                reduce_mode="keep_last",
            ),
            score_limit,
        ],
        # We skip MG-ADL and Schwab and England ADL, which are other scales
        exclude=dict(regex=[r"\bmg\b", r"\bengland\b"], window=-4),
    )
]
