from edsnlp.utils.regex import make_pattern

preceding_directions = [
    r"(?P<direction_since>depuis|depuis\s+le|[àa]\s+partir\s+d[eu]|du)",
    r"(?P<direction_ago>il\s+y\s+a)",
    r"(?P<direction_for>pendant|durant|pdt)",
    r"(?P<direction_in>dans)",
    r"(?P<direction_until>jusqu'[àa]u?|au)",
]

following_directions = [
    r"(?P<direction_in>prochaine?s?|suivante?s?)",
    r"(?P<direction_ago>derni[eè]re?s?)",
]

preceding_direction_pattern = make_pattern(preceding_directions, with_breaks=True)
following_direction_pattern = make_pattern(following_directions, with_breaks=True)
