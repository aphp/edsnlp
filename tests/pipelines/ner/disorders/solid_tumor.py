results_solid_tumor = dict(
    has_match=[True, True, False, True, True, True, True, True, True],
    detailled_status=[
        "LOCALIZED",
        "LOCALIZED",
        None,
        "METASTASIS",
        "METASTASIS",
        "LOCALIZED",
        "METASTASIS",
        "METASTASIS",
        "METASTASIS",
        "METASTASIS",
    ],
    assign=None,
    texts=[
        "Présence d'un carcinome intra-hépatique.",
        "Patient avec un K sein.",
        "Il y a une tumeur bénigne",
        "Tumeur métastasée",
        "Cancer du poumon au stade 4",
        "Cancer du poumon au stade 2",
        "Présence de nombreuses lésions secondaires",
        "Patient avec fracture abcddd secondaire. Cancer de",
        "Patient avec lesions non ciblées",
        "TNM: pTx N1 M1",
    ],
)

solid_tumor_config = dict(use_patterns_metastasis_ct_scan=True, use_tnm=True)
