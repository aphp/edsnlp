main_pattern = dict(
    source="main",
    regex=[
        r"lymphom(?:.{1,10}hodgkin)",
        r"lymphom",
        r"lymphangio",
        r"sezary",
        r"burkitt",
        r"kaposi",
        r"hodgkin",
        r"amylose",
        r"plasm[ao]cytome",
        r"lympho.{1,3}sarcome",
        r"lympho.?prolif",
        r"hemopathie.{1,10}lymphoide",
        r"macroglobulinemie",
        r"immunocytome",
        r"maladie.des.chaine",
        r"histiocytose.{1,5}(maligne|langerhans)",
        r"waldenst(ro|or)m",
        r"mycos.{1,10}fongoide",
        r"myelome",
        r"maladie.{1,5}immunoproliferative.{1,5}maligne",
        r"leucemie.{1,10}plasmocyte",
    ],
    regex_attr="NORM",
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bLNH\b",
        r"\bLH\b",
        r"\bEATL\b",
        r"\bLAGC\b",
        r"\bLDGCB\b",
    ],
    regex_attr="TEXT",
    exclude=dict(
        regex=["/L", "/mL"],
        window=10,
    ),
)


gammapathy = dict(
    source="gammapathy",
    regex=[
        r"gammapathie monoclonale",
    ],
    exclude=dict(
        regex=[
            "benin",
            "benign",
            "signification.indeter",
            "NMSI",
            "MGUS",
        ],
        window=(0, 5),
    ),
    regex_attr="NORM",
)


default_patterns = [
    main_pattern,
    acronym,
    # gammapathy,
]
