from ...utils import normalize_space_characters
from ..utils import float_regex, other_unit

# Speed in metres or centimetres per second, where `ms` is a shorthand for m/s
speed = rf"(?:{float_regex})\s*(?:c?m\s*/\s*s(?:ec(?:onde)?s?)?|ms\b)?"

default_patterns = normalize_space_characters(
    [
        dict(
            source="gait_speed",
            regex=[
                "vitesse de marche",
            ],
            assign=[
                dict(
                    name="value",
                    # We skip durations such as `6 s` in `4 m en 6 s`
                    regex=(
                        rf"(?<![\w/.,+-])({speed}){other_unit}"
                        r"(?!\s*(?:s|sec\w*|min\w*|mn)\b)"
                    ),
                    window=(0, 7),
                    reduce_mode="keep_first",
                ),
            ],
        )
    ]
)
