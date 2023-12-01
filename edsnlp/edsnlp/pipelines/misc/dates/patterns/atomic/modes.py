from edsnlp.utils.regex import make_pattern

modes = [
    r"(?P<mode_FROM>depuis|depuis\s+le|[àa]\s+partir\s+d[eu]|du)",
    r"(?P<mode_UNTIL>jusqu'[àa]u?|au)",
]

mode_pattern = make_pattern(modes, with_breaks=True)
