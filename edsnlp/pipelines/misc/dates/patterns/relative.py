from edsnlp.utils.regex import make_pattern

from . import directions, numbers, units


def make_specific_pattern(name, pattern, forward: bool = True):

    if forward:
        p = directions.preceding_direction_pattern
        p += r"\s+"
        p += make_pattern(numbers.numbers, name=name)
        p += r"\s+"
        p += pattern
    else:
        p = make_pattern(numbers.numbers, name=name)
        p += r"\s+"
        p += pattern
        p += r"\s+"
        p += directions.following_direction_pattern

    return p


relative_patterns = [
    make_specific_pattern(name=name, pattern=pattern, forward=forward)
    for forward in [True, False]
    for name, pattern in units.patterns.items()
]

specific = {
    r"hier": dict(direction="ago", day=1),
    r"avant[-\s]hier": dict(direction="ago", day=2),
    r"demain": dict(direction="in", day=1),
    r"après[-\s]demain": dict(direction="in", day=2),
}

relative_patterns.append(make_pattern(list(specific.keys()), name="specific"))
