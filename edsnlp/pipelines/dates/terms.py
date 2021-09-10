from typing import List

numeric_dates: List[str] = [
    r"[0123]?\d[\/\.\-][01]?\d[\/\.\-](?:19\d\d|20[012]\d|\d\d)",
]

relative_dates: List[str] = [
    r"(?:avant\-)?hier",
    r"(?:après )?demain",
    r"l['ae] ?(?:semaine|année|mois) derni[èe]re?",
    r"l['ae] ?(?:semaine|année|mois) prochaine?",
    r"il y a .{,10} (?:heures?|jours?|mois|années?)",
]

dates = numeric_dates + relative_dates
