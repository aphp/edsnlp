from edsnlp.utils.regex import make_pattern

modifiers = [
    r"(depuis\s+le|[àa]\s+partir\s+du)",
    r"jusqu'au",
]

modifier_pattern = make_pattern(modifiers, name="modifier")
