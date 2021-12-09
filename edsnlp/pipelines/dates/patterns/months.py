from typing import List

letter_months: List[str] = [
    r"janvier",
    r"janv\.?",
    r"f[ée]vrier",
    r"f[ée]v\.?",
    r"mars",
    r"mar\.?",
    r"avril",
    r"avr\.?",
    r"mai",
    r"juin",
    r"juillet",
    r"juill?\.?",
    r"ao[uû]t",
    r"septembre",
    r"sept?\.?",
    r"octobre",
    r"oct\.?",
    r"novembre",
    r"nov\.",
    r"d[ée]cembre",
    r"d[ée]c\.?",
]

letter_month_pattern = r"\b(" + "|".join(letter_months) + r")\b"

numeric_month_pattern = r"(?<!\d)(0?[1-9]|1[0-2])(?!\d)"
numeric_month_pattern_with_leading_zero = r"(?<!\d)(0[1-9]|1[0-2])(?!\d)"

month_pattern = f"({letter_month_pattern}|{numeric_month_pattern})"
