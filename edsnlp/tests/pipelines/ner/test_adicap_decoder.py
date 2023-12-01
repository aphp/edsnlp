examples = [
    "1. Codification ADICAP : OHNP6071",
    "2. Codification ADICAP : CSGS7624",
    "3. Codification ADICAP : BSDETMA0",
    "4. Codification ADICAP : BSSG0503",
    "5. Codification ADICAP : BSGS0503",
    "6. Codification ADICAP : HSGS0G26",
    "7. Codification ADICAP : HSGS0150",
]


expected_outputs = [
    {
        "code": "OHNP6071",
        "sampling_mode": "PIECE OPERATOIRE AVEC EXERESE COMPLETE DE L'ORGANE",
        "technic": "HISTOLOGIE ET CYTOLOGIE PAR INCLUSION",
        "organ": "SYSTEME NERVEUX PERIPHERIQUE",
        "pathology": "PATHOLOGIE GÉNÉRALE NON TUMORALE",
        "pathology_type": (
            "THROMBOCYTOPENIE - HYPOPLASIE ACQUISE "
            "MEGACARYOCYTAIRE ET PLAQUETTAIRE (SAI)"
        ),
        "behaviour_type": "CARACTERES GENERAUX",
    },
    {
        "code": "CSGS7624",
        "sampling_mode": "CYTOPONCTION NON GUIDEE PAR IMAGERIE",
        "technic": "COLORATION SPECIALE - HISTO ET CYTOCHIMIE",
        "organ": "SEIN (ÉGALEMENT UTILISÉ CHEZ L'HOMME)",
        "pathology": "PATHOLOGIE GÉNÉRALE NON TUMORALE",
        "pathology_type": (
            "INFLAMMATION SUBAIGUE ET CHRONIQUE LEGERE TOXIQUE (ALCOOL, ETC)"
        ),
        "behaviour_type": (
            "MODIFICATION DE VOLUME, DYSTROPHIE, "
            "KYSTE, METAPLASIE ET DYSPLASIE ACQUISE"
        ),
    },
    {
        "code": "BSDETMA0",
        "sampling_mode": "BIOPSIE CHIRURGICALE",
        "technic": "COLORATION SPECIALE - HISTO ET CYTOCHIMIE",
        "organ": "ESTOMAC",
        "pathology": "PATHOLOGIE TUMORALE",
        "pathology_type": "THYMOME AVEC EMBOLES NEOPLASIQUES",
        "behaviour_type": "CANCER MÉTASTATIQUE",
    },
    {
        "code": "BSSG0503",
        "sampling_mode": "BIOPSIE CHIRURGICALE",
        "technic": "COLORATION SPECIALE - HISTO ET CYTOCHIMIE",
        "organ": "GANGLION LYMPHATIQUE",
        "pathology": "PATHOLOGIE PARTICULIERE DES ORGANES",
        "pathology_type": "HISTIOCYTOSE EN AMAS GANGLIONNAIRE",
        "behaviour_type": None,
    },
    {
        "code": "BSGS0503",
        "sampling_mode": "BIOPSIE CHIRURGICALE",
        "technic": "COLORATION SPECIALE - HISTO ET CYTOCHIMIE",
        "organ": "SEIN (ÉGALEMENT UTILISÉ CHEZ L'HOMME)",
        "pathology": None,
        "pathology_type": None,
        "behaviour_type": None,
    },
    {
        "code": "HSGS0G26",
        "sampling_mode": "HISTOPONCTION GUIDEE PAR IMAGERIE",
        "technic": "COLORATION SPECIALE - HISTO ET CYTOCHIMIE",
        "organ": "SEIN (ÉGALEMENT UTILISÉ CHEZ L'HOMME)",
        "pathology": "CYTOPATHOLOGIE",
        "pathology_type": (
            "CELLULES GLANDULAIRES ATYPIQUES "
            "EN FAVEUR D'UNE NATURE NEOPLASIQUE - SAI"
        ),
        "behaviour_type": "MATERIEL DE SIGNIFICATION INDETERMINEE",
    },
    {
        "code": "HSGS0150",
        "sampling_mode": "HISTOPONCTION GUIDEE PAR IMAGERIE",
        "technic": "COLORATION SPECIALE - HISTO ET CYTOCHIMIE",
        "organ": "SEIN (ÉGALEMENT UTILISÉ CHEZ L'HOMME)",
        "pathology": "PATHOLOGIE GÉNÉRALE NON TUMORALE",
        "pathology_type": "LESION D'INTERET PARTICULIER",
        "behaviour_type": "MALADIE INNEE ET GRANDE MALFORMATION EXTERNE",
    },
]


def test_scores(blank_nlp):

    blank_nlp.add_pipe("eds.adicap")

    for input, expected in zip(examples, expected_outputs):
        doc = blank_nlp(input)
        assert len(doc.ents) > 0
        assert doc.ents[0]._.adicap.dict() == expected
