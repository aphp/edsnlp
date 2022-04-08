from edsnlp.utils.regex import make_pattern

units = [
    r"(?P<unit_year>ans?|ann[ée]es?)",
    r"(?P<unit_semester>semestres?)",
    r"(?P<unit_trimester>trimestres?)",
    r"(?P<unit_month>mois)",
    r"(?P<unit_week>semaines?)",
    r"(?P<unit_day>jours?|journ[ée]es?)",
    r"(?P<unit_hour>h|heures?)",
    r"(?P<unit_minute>min|minutes?)",
    r"(?P<unit_second>sec|secondes?|s)",
]

unit_pattern = make_pattern(units, with_breaks=True)
