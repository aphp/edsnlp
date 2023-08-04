from edsnlp.utils.regex import make_pattern

preceding_directions = [
    r"(?P<direction_past>depuis|depuis\s+le|il\s+y\s+a|à)",
    r"(?P<direction_future>dans)",
]

following_directions = [
    r"(?P<direction_future>prochaine?s?|suivante?s?|plus\s+tard)",
    r"(?P<direction_past>derni[eè]re?s?|passée?s?|pr[ée]c[ée]dente?s?|plus\s+t[ôo]t)",
]

preceding_direction_pattern = make_pattern(preceding_directions, with_breaks=True)
following_direction_pattern = make_pattern(following_directions, with_breaks=True)
