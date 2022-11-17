BENINE = r"benign|benin|(grade.?\b[i1]\b)"
STAGE = r"stade ([^\s]*)"

main_pattern = dict(
    source="main",
    regex=[
        r"carcinom(?!.{0,10}in.?situ)",
        r"seminome",
        r"(?<!lympho)(?<!lympho-)sarcome",
        r"blastome",
        r"cancer([^o]|\s|\b)",
        r"adamantinome",
        r"chordome",
        r"craniopharyngiome",
        r"melanome",
        r"neoplas(?!tine)",
        r"linite",
        r"melanome",
        r"mesoteliome",
        r"mesotheliome",
        r"seminome",
        r"myxome",
        r"paragangliome",
        r"craniopharyngiome",
        r"k .{0,5}(prostate|sein)",
        r"pancoast.?tobias",
        r"syndrome.{1,10}lynch",
        r"li.?fraumeni",
        r"paraneoplasiqu",
        r"germinome",
        r"adeno[\s-]?k",
        r"thymome",
        r"\bnut\b",
        r"\bgist\b",
        r"\bchc\b",
        r"\badk\b",
        r"\btves\b",
        r"tumeur",
        r"carcinoid",
        r"histiocytome",
        r"ependymome",
        r"primitif",
    ],
    exclude=dict(
        regex=BENINE,
        window=(0, 5),
    ),
    regex_attr="NORM",
    assign=[
        dict(
            name="metastasis",
            regex=r"(metasta|multinodul)",
            window=(-3, 7),
            reduce_mode="keep_last",
        ),
        dict(
            name="stage",
            regex=STAGE,
            window=7,
            reduce_mode="keep_last",
        ),
    ],
)

metastasis_pattern = dict(
    source="metastasis",
    regex=[
        r"cellules tumorales circulantes",
        r"metasta",
        r"multinodul",
        r"carcinose",
        r"ruptures? corticale",
        r"envahissement.{0,15}parties\smolle",
        r"(localisation|lesion)s? .{0,20}second",
        r"(lymphangite|meningite).{1,5}carcinomateuse",
    ],
    regex_attr="NORM",
)

default_patterns = [
    main_pattern,
    metastasis_pattern,
]
