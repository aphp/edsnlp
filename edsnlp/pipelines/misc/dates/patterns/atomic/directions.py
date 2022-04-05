from edsnlp.utils.regex import make_pattern

preceding_directions = [
    r"(?P<direction_since>depuis|depuis\s+le)",
    r"(?P<direction_after>[àa]\s+partir\s+d[eu]|du)",
    r"(?P<direction_past>il\s+y\s+a)",
    r"(?P<direction_during>pendant|durant|pdt)",
    r"(?P<direction_future>dans)",
    r"(?P<direction_until>jusqu'[àa]u?|au)",
]

following_directions = [
    r"(?P<direction_future>prochaine?s?|suivante?s?)",
    r"(?P<direction_past>derni[eè]re?s?|passée?s?)",
]

preceding_direction_pattern = make_pattern(preceding_directions, with_breaks=True)
following_direction_pattern = make_pattern(following_directions, with_breaks=True)
