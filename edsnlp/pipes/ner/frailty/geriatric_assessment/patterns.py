from ..utils import make_assign_regex, normalize_space_characters

other = dict(
    source="other",
    regex=[
        "(?:sur le )?plan geriatrique",
    ],
    regex_attr="NORM",
)

onco_geriatry = dict(
    source="other_oncogeriatry",
    regex=[r"(?:onco(?:[\s-]+)?)?geriatr(?:ique|e)"],
    regex_attr="NORM",
    assign=[
        dict(
            name="complements",
            regex=make_assign_regex(["evaluation", "avis"]),
            window=-3,
            required=True,
        )
    ],
)

default_patterns = normalize_space_characters([other, onco_geriatry])
