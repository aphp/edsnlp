from edsnlp.utils.regex import make_pattern

from .atomic import directions, numbers, units
from .atomic.modes import mode_pattern
from .current import current_pattern


def make_specific_pattern(mode: str = "forward"):

    if mode == "forward":
        p = directions.preceding_direction_pattern
        p += r"\s+"
        p += numbers.number_pattern
        p += r"\s*"
        p += units.unit_pattern
    elif mode == "backward":
        p = numbers.number_pattern
        p += r"\s*"
        p += units.unit_pattern
        p += r"\s+"
        p += directions.following_direction_pattern
    else:
        p = directions.preceding_direction_pattern
        p += r"\s+"
        p += numbers.number_pattern
        p += r"\s*"
        p += units.unit_pattern
        p += r"\s+"
        p += directions.following_direction_pattern

    return p


specific = {
    "minus1": (r"hier", dict(direction="PAST", day=1)),
    "minus2": (r"avant[-\s]hier", dict(direction="PAST", day=2)),
    "plus1": (r"demain", dict(direction="FUTURE", day=1)),
    "plus2": (r"après[-\s]demain", dict(direction="FUTURE", day=2)),
}

specific_pattern = make_pattern(
    [f"(?P<specific_{k}>{p})" for k, (p, _) in specific.items()],
)

specific_dict = {k: v for k, (_, v) in specific.items()}

relative_pattern = [
    make_specific_pattern(mode="forward"),
    make_specific_pattern(mode="backward"),
    make_specific_pattern(mode="all"),
    specific_pattern,
    current_pattern,
]

relative_pattern = [r"(?<=" + mode_pattern + r".{,3})?" + p for p in relative_pattern]
