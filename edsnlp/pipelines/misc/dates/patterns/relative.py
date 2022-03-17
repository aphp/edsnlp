h_to_y = r"(?P<unit>heures?|jours?|semaines?|mois|ann[ée]es?|ans?)"
w_to_y = r"(?P<unit>semaine|ann[ée]e|an|mois)"

relative_patterns = {
    "ago": dict(
        pattern=r"il\s+y\s+a\s+(?P<value>.{,10}?)\s+" + h_to_y,
        direction="before",
    ),
    "in": dict(
        pattern=r"dans\s+(?P<value>.{,10}?)\s+" + h_to_y,
        direction="after",
    ),
    "last": dict(
        pattern=r"l['ae]\s*" + w_to_y + r"\s+derni[èe]re?",
        value=1,
        direction="before",
    ),
    "next": dict(
        pattern=r"l['ae]\s*" + w_to_y + r"\s+prochaine?",
        value=1,
        direction="after",
    ),
    "since": dict(
        pattern=r"depuis(?!\s+l'[aâ]ge)\s*(?P<value>.{,10})\s+"
        + h_to_y
        + r"(?!\s+derni[èe]re?)",
        direction="before",
    ),
    "during": dict(
        pattern=r"(pendant|pdt|pour)\s+(?!dans)(?P<value>.{,10}?)\s+" + h_to_y,
        direction="unknown",
    ),
    "two_days_ago": dict(
        pattern=r"avant\-?\s*hier",
        unit="day",
        value=2,
        direction="before",
    ),
    "one_day_ago": dict(
        pattern=r"(?<!avant)(?<!avant[\- ])hier",
        unit="day",
        value=1,
        direction="before",
    ),
    "two_days_after": dict(
        pattern=r"apr[èe]s\-?\s*demain",
        unit="day",
        value=2,
        direction="after",
    ),
    "one_day_after": dict(
        pattern=r"(?<!apr[èe]s)(?<!apr[èe]s[\- ])demain",
        unit="day",
        value=1,
        direction="after",
    ),
    #     spans can only be matched once,
    #     the patterns are used in the same order as the dict.
    #     we can add wide debug patterns at the end to see what isn't matched
    #     "debug":dict(
    #         pattern=h_to_y,
    #         unit="a",
    #         value=0,
    #         direction="a",)
}
