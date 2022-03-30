from edsnlp.utils.regex import make_pattern

year = [
    r"ans?",
    r"ann[ée]es?",
]

semester = [
    r"semestres?",
]

trimester = [
    r"trimestres?",
]

month = [
    r"mois",
]

week = [
    r"semaines?",
]

day = [
    r"jours?",
    r"journ[ée]s?",
]

hour = [
    r"heures?",
    r"h",
]

minute = [
    r"minutes?",
    r"min",
]

second = [
    r"secondes?",
    r"secs?",
]


patterns = dict(
    year=make_pattern(year),
    semester=make_pattern(semester),
    trimester=make_pattern(trimester),
    month=make_pattern(month),
    week=make_pattern(week),
    day=make_pattern(day),
    hour=make_pattern(hour),
    minute=make_pattern(minute),
    second=make_pattern(second),
)
