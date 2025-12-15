from ..utils import make_status_assign

other = dict(
    source="other",
    regex=[
        "(?:sur le )?plan antalgique",
    ],
    regex_attr="NORM",
)
healthy = dict(
    source="healthy",
    regex=[r"indolores?"],
    regex_attr="NORM",
)

altered = dict(
    source="altered",
    regex=[
        "douleurs?",
        r"douleurs? (?:\b\w+\b\s){0,5}mecaniques?",
        r"douloureu(?:x|ses?)",
        r"(?:traitements? )?antalgiques?",
        "souffrances?",
        "neuropathie",
        r"\balgique",
        "antalgie",
    ],
    regex_attr="NORM",
    exclude=dict(regex="abdominale?s?", window=2),
)
mild = dict(source="mild", regex=[r"pall?iers? (?:1|2|i(?:i)?)"], regex_attr="NORM")

severe = dict(
    source="severe",
    regex=["hyperalgie", "etat de souffrance", r"pall?iers? (?:3|iii)"],
    regex_attr="NORM",
)

status = dict(
    source="other_status",
    regex=["etat (?:ant)?algique", "statut (?:ant)?algique"],
    regex_attr="NORM",
    assign=make_status_assign(),
)

default_patterns = [other, healthy, altered, mild, severe, status]
