from edsnlp.utils.regex import make_pattern

ago_pattern = r"il\s+y\s+a\s+.{,10}?\s+(heures?|jours?|semaines?|mois|ann[ée]es?|ans?)"
in_pattern = r"dans\s+.{,10}?\s+(heures?|jours?|semaines?|mois|ann[ée]es?|ans?)"
last_pattern = r"l['ae]\s*(semaine|année|an|mois)\s+derni[èe]re?"
next_pattern = r"l['ae]\s*(semaine|année|an|mois)\s+prochaine?"

# "depuis" is not recognized
since_pattern = (
    r"(?<=depuis\s)\s*.{,10}\s+(heures?|jours?|semaines?"
    r"|mois|ann[ée]es?|ans?)(\s+derni[èe]re?)?"
)
during_pattern = r"(pendant|pdt|pour)\s+.{,10}?\s+(heures?|jours?|mois|ann[ée]es?|ans?)"

week_patterns = [
    r"(avant\-?\s*)?hier",
    r"(apr[èe]s\-?\s*)?demain",
]
week_pattern = make_pattern(week_patterns, with_breaks=True)

relative_pattern = make_pattern(
    patterns=[
        ago_pattern,
        in_pattern,
        last_pattern,
        next_pattern,
        since_pattern,
        week_pattern,
    ],
    with_breaks=True,
)
