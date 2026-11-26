from ..utils import make_assign_regex, make_status_assign, normalize_space_characters

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
    exclude=dict(regex=make_assign_regex(["abdominale?s?", "abdomen"]), window=(-4, 2)),
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
    exclude=dict(
        regex=make_assign_regex(["abdominale?s?", "aigue?s?"]), window=(-4, 2)
    ),
)
# The boundaries keep `palier 10` and `palier 1.5` out of the analgesic levels
mild = dict(
    source="mild", regex=[r"\bpall?iers? (?:1|2|ii?)\b(?![.,]\d)"], regex_attr="NORM"
)

severe = dict(
    source="severe",
    regex=["hyperalgie", "etat de souffrance", r"\bpall?iers? (?:3|iii)\b(?![.,]\d)"],
    regex_attr="NORM",
)

status = dict(
    source="other_status",
    regex=["etat (?:ant)?algique", "statut (?:ant)?algique"],
    regex_attr="NORM",
    assign=make_status_assign(),
)

default_patterns = normalize_space_characters(
    [other, healthy, altered, mild, severe, status]
)
