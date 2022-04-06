from edsnlp.utils.regex import make_pattern

preceding_directions = [
    r"(?P<direction_PAST>depuis|depuis\s+le|il\s+y\s+a)",
    r"(?P<direction_FUTURE>dans)",
]

following_directions = [
    r"(?P<direction_FUTURE>prochaine?s?|suivante?s?)",
    r"(?P<direction_PAST>derni[eè]re?s?|passée?s?)",
]

preceding_direction_pattern = make_pattern(preceding_directions, with_breaks=True)
following_direction_pattern = make_pattern(following_directions, with_breaks=True)
