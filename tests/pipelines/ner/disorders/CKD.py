results_ckd = dict(
    has_match=[
        True,
        False,
        True,
        True,
        False,
        True,
        False,
        True,
        True,
        True,
        False,
    ],
    detailled_status="PRESENT",
    assign=8 * [None] + [{"stage": "IV"}, {"dfg": 30}, None],
    texts=[
        "Patient atteint d'une glomérulopathie.",
        "Patient atteint d'une tubulopathie aigüe.",
        "Patient transplanté rénal",
        "Présence d'une insuffisance rénale aigüe sur chronique",
        "Le patient a été dialysé",  # ponctuelle
        "Le patient est dialysé chaque lundi",  # chronique
        "Présence d'une IRC",  # severity non mentionned
        "Présence d'une IRC sévère",
        "Présence d'une IRC de classe IV",
        "Présence d'une IRC avec DFG à 30",  # severe
        "Présence d'une maladie rénale avec DFG à 110",  # no renal failure
    ],
)
