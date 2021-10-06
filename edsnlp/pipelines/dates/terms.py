from typing import List

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
    "premier",
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
    r"[0123]?\d[\/\.\-\s][01]?\d[\/\.\-\s](?:19\d\d|20[012]\d|\d\d)",
    r"(?:19\d\d|20[012]\d|\d\d)[\/\.\-\s][01]?\d[\/\.\-\s][0123]?\d",
]

text_dates: List[str] = [
    r"(?:depuis|en)\s*" + month_pattern + r"?\s+(?:19\d\d|20[012]\d|\d\d)",
    r"[0123]?\d\d+\s*" + month_pattern + r"\s+(?:19\d\d|20[012]\d|\d\d)",
    day_pattern + r"\s+" + month_pattern + r"\s+(?:19\d\d|20[012]\d|\d\d)",
]

unknown_year: List[str] = [
    r"[0123]?\d[\/\.\-\s][01]\d",
    r"(?:depuis|en)\s+" + month_pattern,
    r"[0123]?\d\s*" + month_pattern,
    day_pattern + r"\s+" + month_pattern,
]

relative_expressions: List[str] = [
    r"(?:avant\-)?hier",
    r"(?:apr[èe]s )?demain",
    r"l['ae] ?(?:semaine|année|an|mois) derni[èe]re?",
    r"l['ae] ?(?:semaine|année|an|mois) prochaine?",
    r"il y a .{,10} (?:heures?|jours?|mois|ann[ée]es?|ans?)",
    r"depuis .{,10} (?:heures?|jours?|mois|ann[ée]es?|ans?)",
    r"dans .{,10} (?:heures?|jours?|mois|ann[ée]es?|ans?)",
]

no_year = "|".join(unknown_year)
absolute = "|".join(numeric_dates + text_dates)
relative = "|".join(relative_expressions)
