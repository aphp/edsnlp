from edsnlp.utils.regex import make_pattern

letter_months = [
    r"(?P<month_01>janvier|janv\.?)",
    r"(?P<month_02>f[ée]vrier|f[ée]v\.?)",
    r"(?P<month_03>mars|mar\.?)",
    r"(?P<month_04>avril|avr\.?)",
    r"(?P<month_05>mai)",
    r"(?P<month_06>juin)",
    r"(?P<month_07>juillet|juill?\.?)",
    r"(?P<month_08>ao[uû]t)",
    r"(?P<month_09>septembre|sept?\.?)",
    r"(?P<month_10>octobre|oct\.?)",
    r"(?P<month_11>novembre|nov\.?)",
    r"(?P<month_12>d[ée]cembre|d[ée]c\.?)",
]


letter_month_pattern = make_pattern(letter_months, with_breaks=True)

numeric_month_pattern = r"(?<!\d)(0?[1-9]|1[0-2])(?!\d)"
lz_numeric_month_pattern = r"(?<!\d)(0[1-9]|1[0-2])(?!\d)"

numeric_month_pattern = f"(?P<month>{numeric_month_pattern})"
lz_numeric_month_pattern = f"(?P<month>{lz_numeric_month_pattern})"
month_pattern = f"({letter_month_pattern}|{numeric_month_pattern})"
