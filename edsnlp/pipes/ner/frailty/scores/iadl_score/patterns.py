from ..utils import assigned_score_value, score_limit

# We skip item values, e.g. `telephone 1`, since they are not the IADL total
iadl_value = (
    r"(?<!\b(?:telephone|transports?|medicaments?|budget|courses|repas|menage|"
    r"linge|lessive|finances?)\s*[:=]?\s*)" + assigned_score_value
)

default_patterns = [
    dict(
        source="iadl",
        regex=r"\biadl\b",
        assign=[
            dict(
                name="value",
                regex=iadl_value,
                window=(0, 35),
                replace_entity=False,
                reduce_mode="keep_last",
            ),
            score_limit,
        ],
    )
]
