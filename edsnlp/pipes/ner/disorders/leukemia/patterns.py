main_pattern = dict(
    source="main",
    regex=[
        r"leucemie",
        r"(syndrome.)?myeloproliferatif",
        r"m[yi]eloprolifer",
    ],
    exclude=dict(
        regex=[
            "plasmocyte",
            "benin",
            "benign",
        ],
        window=5,
    ),
    regex_attr="NORM",
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bLAM\b",
        r"\bLAM.?[0-9]",
        r"\bLAL\b",
        r"\bLMC\b",
        r"\bLCE\b",
        r"\bLMM[JC]\b",
        r"\bLCN\b",
        r"\bAREB\b",
        r"\bAPMF\b",
        r"\bLLC\b",
        r"\bSMD\b",
        r"LA my[éèe]lomonocytaire",
    ],
    regex_attr="TEXT",
    exclude=dict(
        regex="anti",
        window=-20,
    ),
)

other = dict(
    source="other",
    regex=[
        r"myelofibrose",
        r"vaquez",
        r"thrombocytemie.{1,3}essentielle",
        r"splenomegalie.{1,3}myeloide",
        r"mastocytose.{1,5}maligne",
        r"polyglobulie.{1,10}essentielle",
        r"letterer.?siwe",
        r"anemie.refractaire.{1,20}blaste",
        r"m[iy]elod[iy]splasi",
        r"syndrome.myelo.?dysplasique",
    ],
    regex_attr="NORM",
)

default_patterns = [
    main_pattern,
    acronym,
    other,
]
