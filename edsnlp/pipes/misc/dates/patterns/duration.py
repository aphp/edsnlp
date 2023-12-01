from .atomic import numbers, units

cue_pattern = r"(pendant|durant|pdt)"

duration_pattern = [
    cue_pattern + r".{,3}" + numbers.number_pattern + r"\s*" + units.unit_pattern
]
