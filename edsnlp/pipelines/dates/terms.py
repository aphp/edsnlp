from typing import List

months: List[str] = [
    r"janvier",
    r"janv\.?",
    r"f[ée]vrier",
    r"f[ée]v\.?",
    r"mars",
    r"avril",
    r"avr\.?",
    r"mai",
    r"juin",
    r"juillet",
    r"juill?\.?",
    r"ao[uû]t",
    r"septembre",
    r"sept\.?",
    r"octobre",
    r"oct\.?",
    r"novembre",
    r"nov\.",
    r"d[ée]cembre",
    r"d[ée]c\.?",
]
month_pattern = "(?:" + "|".join(months) + ")"

numeric_dates: List[str] = [
    r"[0123]?\d[\/\.\-\s][01]?\d[\/\.\-\s](?:19\d\d|20[012]\d|\d\d)",
    r"(?:19\d\d|20[012]\d|\d\d)[\/\.\-\s][01]?\d[\/\.\-\s][0123]?\d",
]

text_dates: List[str] = [
    r"(?:depuis|en)\s*" + month_pattern + r"?\s+(?:19\d\d|20[012]\d|\d\d)",
    r"[0123]?\d\d+\s*" + month_pattern + r"\s+(?:19\d\d|20[012]\d|\d\d)",
]

unknown_year: List[str] = [
    r"[0123]?\d[\/\.\-\s][01]\d",
    r"(?:depuis|en)\s+" + month_pattern,
    r"[0123]?\d\s*" + month_pattern,
]

relative_expressions: List[str] = [
    r"(?:avant\-)?hier",
    r"(?:après )?demain",
    r"l['ae] ?(?:semaine|année|mois) derni[èe]re?",
    r"l['ae] ?(?:semaine|année|mois) prochaine?",
    r"il y a .{,10} (?:heures?|jours?|mois|années?)",
    r"depuis .{,10} (?:heures?|jours?|mois|années?)",
]

no_year = "|".join(unknown_year)
absolute = "|".join(numeric_dates + text_dates)
relative = "|".join(relative_expressions)
