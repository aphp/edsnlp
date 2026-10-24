results_adl = [
    ("ADL 6/6 IADL 3/4. Le patient est en EHPAD.", (6.0, "healthy")),
    ("ADL 6/8 IADL 3/4.", None),
    ("ADL 5/6 IADL 3/4.", (5.0, "altered_nondescript")),
    ("ADL 5 IADL 3/4.", (5.0, "altered_nondescript")),
    (
        "ADL (hygiène corporelle 0.5, habillage 0.5, aller aux toilettes 1, locomotion 1, continence 1, repas 1): 5/6 IADL 4/4",  # noqa E501
        (5.0, "altered_nondescript"),
    ),
    ("ADL 0", (0.0, "altered_severe")),
    ("ADL 5 / 6", (5.0, "altered_nondescript")),
    ("ADL 3/7.5", None),
    ("ADL -2", None),
    ("ADL G6", None),
    ("MG-ADL 6", None),
    ("Schwab et England ADL 80", None),
    ("ADL (habillage 1, repas 1)", None),
    ("ADL : habillage 1, toilette 0,5, total 4/6", (4.0, "altered_nondescript")),
]
