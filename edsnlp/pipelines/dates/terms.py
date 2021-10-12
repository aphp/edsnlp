from typing import List


def add_break(patterns: List[str]):
    return [r"\b" + pattern + r"\b" for pattern in patterns]


months: List[str] = [
    r"[jJ]anvier",
    r"[jJ]anv\.?",
    r"[fF][ée]vrier",
    r"[fF][ée]v\.?",
    r"[mM]ars",
    r"[aA]vril",
    r"[aA]vr\.?",
    r"[mM]ai",
    r"[jJ]uin",
    r"[jJ]uillet",
    r"[jJ]uill?\.?",
    r"[aA]o[uû]t",
    r"[sS]eptembre",
    r"[sS]ept\.?",
    r"[oO]ctobre",
    r"[oO]ct\.?",
    r"[nN]ovembre",
    r"[nN]ov\.",
    r"[dD][ée]cembre",
    r"[dD][ée]c\.?",
]
month_pattern = "(?:" + "|".join(months) + ")"

days: List[str] = [
    r"(?:premier|1\s*er)",
    "deux",
    "trois",
    "quatre",
    "cinq",
    "six",
    "sept",
    "huit",
    "neuf",
    "dix",
    "onze",
    "douze",
    "treize",
    "quatorze",
    "quinze",
    "seize",
    r"dix[-\s]sept",
    r"dix[-\s]huit",
    r"dix[-\s]neuf",
    "vingt",
    r"vingt[-\s]et[-\s]un",
    r"vingt[-\s]deux",
    r"vingt[-\s]trois",
    r"vingt[-\s]quatre",
    r"vingt[-\s]cinq",
    r"vingt[-\s]six",
    r"vingt[-\s]sept",
    r"vingt[-\s]huit",
    r"vingt[-\s]neuf",
    r"trente",
    r"trente[-\s]et[-\s]un",
]
day_pattern = "(?:" + "|".join(days) + ")"

numeric_dates: List[str] = [
    r"(?<!\d)(?:3[01]|[12][0-9]|0?[1-9]|1er)[\/\.\-\s][01]?\d[\/\.\-\s](?:19\d\d|20[012]\d|\d\d)",
]

full_dates: List[str] = [
    r"(?:19\d\d|20[012]\d)[\/\.\-\s][01]?\d[\/\.\-\s](?:3[01]|[12][0-9]|0?[1-9])",
]

text_dates: List[str] = [
    r"(?:depuis|en)\s+" + month_pattern + r"?\s+(?:19\d\d|20[012]\d|\d\d)",
    r"(?<!\d)(?:3[01]|[12][0-9]|0?[1-9]|1er)\s*"
    + month_pattern
    + r"\s+(?:19\d\d|20[012]\d|\d\d)",
    day_pattern + r"\s+" + month_pattern + r"\s+(?:19\d\d|20[012]\d|\d\d)",
]

unknown_year: List[str] = [
    r"(?<!\d)(?:3[01]|[12][0-9]|0?[1-9]|1er)[\/\.\-\s][01]?\d",
    r"(?:depuis|en)\s+" + month_pattern,
    r"(?<!\d)(?:3[01]|[12][0-9]|0?[1-9]|1er)\s*" + month_pattern,
    day_pattern + r"\s+" + month_pattern,
]

year_only: List[str] = [
    r"(?:depuis|en|d[ée]but|fin)\s+(?:19\d\d|20[012]\d|\d\d)",
    r"(?:depuis|en|d[ée]but|fin)\s+(?:d'|de\s+l')ann[ée]\s+(?:19\d\d|20[012]\d|\d\d)",
]

relative_expressions: List[str] = [
    r"(?:avant[-\s])?hier",
    r"(?:apr[èe]s[-\s])?demain",
    r"l['ae]\s*(?:semaine|année|an|mois) derni[èe]re?",
    r"l['ae]\s*(?:semaine|année|an|mois) prochaine?",
    r"il y a .{,10} (?:heures?|jours?|mois|ann[ée]es?|ans?)",
    r"depuis .{,10} (?:heures?|jours?|mois|ann[ée]es?|ans?)",
    r"dans .{,10} (?:heures?|jours?|mois|ann[ée]es?|ans?)",
]

hours: str = r"\d?\d[h:]\d\d"

no_year = "|".join(add_break(unknown_year))
absolute = "|".join(add_break(numeric_dates + text_dates + year_only))
relative = "|".join(add_break(relative_expressions))

no_year = r"(?:" + no_year + r")(?:\s+" + hours + ")?"
absolute = r"(?:" + absolute + r")(?:\s+" + hours + ")?"
relative = r"(?:" + relative + r")(?:\s+" + hours + ")?"

full_date = "|".join(add_break(full_dates))

false_positives = r"(?:\d\d[\s\.\/\-]?){4,}"
