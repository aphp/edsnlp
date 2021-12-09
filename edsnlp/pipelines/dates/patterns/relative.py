from ..utils import make_pattern

ago_pattern = r"il y a .{,10} (?:heures?|jours?|mois|ann[ée]es?|ans?)"
in_pattern = r"dans .{,10} (?:heures?|jours?|mois|ann[ée]es?|ans?)"
last_pattern = r"l['ae]\s*(?:semaine|année|an|mois)\sderni[èe]re?"
next_pattern = r"l['ae]\s*(?:semaine|année|an|mois) prochaine?"

# "depuis" is not recognized
since_pattern = r"(?<=depuis) .{,10} (?:heures?|jours?|mois|ann[ée]es?|ans?)"

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
