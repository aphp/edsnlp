from edsnlp.utils.regex import make_pattern

preceding_directions = [
    r"(?P<direction_since>depuis)",
    r"(?P<direction_ago>il\s+y\s+a)",
    r"(?P<direction_for>pendant|durant)",
    r"(?P<direction_in>dans)",
    r"(?P<direction_from>[àa]\s+partir\s+de)",
]

following_directions = [
    r"(?P<direction_in>prochaine?s?|suivante?s?)",
    r"(?P<direction_ago>derni[eè]re?s?)",
]

preceding_direction_pattern = make_pattern(preceding_directions)
following_direction_pattern = make_pattern(following_directions)
