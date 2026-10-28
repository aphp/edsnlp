results_iadl = [
    ("ADL 6/6 IADL 3/4. Le patient est en EHPAD.", (3.0, "altered_nondescript")),
    ("ADL 6/8 IADL 3/13.", None),
    ("IADL 3/4 ADL 5/6.", (3.0, "altered_nondescript")),
    ("ADL 5 IADL 4/4.", (4.0, "healthy")),
    ("ADL 5 IADL 4/5.", (4.0, "altered_nondescript")),
    ("ADL 5 IADL 11/11.", (11.0, "healthy")),
    (
        "ADL (hygiène corporelle 0.5, habillage 0.5, aller aux toilettes 1, locomotion 1, continence 1, repas 1): 5/6 IADL 4/4",  # noqa E501
        (4.0, "healthy"),
    ),
    (
        """IADL (aptitude à utiliser le téléphone 1, moyens de transport 0, responsabilité à l'égard de son
traitement 0.5, aptitude à gérer son budget 0): 1.5/4""",  # noqa E501
        (1.5, "altered_nondescript"),
    ),
]
