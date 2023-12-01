from edsnlp.utils.regex import make_pattern

modes = [
    r"(?P<bound_from>depuis|depuis\s+le|[àa]\s+partir\s+d[eu]|du)",
    r"(?P<bound_until>jusqu'[àa]u?|au)",
]

mode_pattern = make_pattern(modes, with_breaks=True)
