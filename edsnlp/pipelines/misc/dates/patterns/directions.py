from edsnlp.utils.regex import make_pattern

preceding_directions = {
    r"depuis": "since",
    r"il\s+y\s+a": "ago",
    r"(pendant|durant)": "for",
    r"dans": "in",
}

following_directions = {
    r"prochaine?s?": "in",
    r"suivante?s?": "in",
    r"derni[eè]re?s?": "ago",
}

preceding_direction_pattern = make_pattern(
    list(preceding_directions.keys()),
    name="direction",
)
following_direction_pattern = make_pattern(
    list(following_directions.keys()),
    name="direction",
)

directions = dict(
    **preceding_directions,
    **following_directions,
)
