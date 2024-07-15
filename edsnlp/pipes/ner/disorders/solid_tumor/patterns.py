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
        r"neoplasie",
        r"neoplasme",
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
        r"germinome",
        r"adeno[\s-]?k",
        r"thymome",
        r"\bnut\b",
        r"\bgist\b",
        r"\bchc\b",
        r"\badk\b",
        r"\btves\b",
        r"\btv.tves\b",
        r"lesion.{1,20}tumor",
        r"tumeur",
        r"carcinoid",
        r"histiocytome",
        r"ependymome",
        # r"primitif", Trop de FP
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
        r"cellule.{1,5}tumorale.{1,5}circulantes",
        r"metasta",
        r"multinodul",
        r"carcinose",
        r"ruptures.{1,5}corticale",
        r"envahissement.{0,15}parties\smolle",
        r"(localisation|lesion)s?.{0,20}second",
        r"(lymphangite|meningite).{1,5}carcinomateuse",
    ],
    regex_attr="NORM",
    exclude=dict(
        regex=r"goitre",
        window=-3,
    ),
)

# Patterns developed for CT-Scan reports
metastasis_ct_scan = dict(
    source="metastasis_ct_scan",
    regex=[
        r"(?i)(m[ée]tasta(se|tique)s?)",
        r"(diss[ée]min[ée]e?s?)",
        r"(carcinose)",
        r"(((allure|l[ée]sion|localisation|progression)s?\s)(suspecte?s?)?.{0,50}(secondaire)s?)",
        r"(l(a|â)ch(é|e|er)\sde\sballons?)",
        r"(l[ée]sions?\s(non\s)?cibles?)",
        r"(rupture.{1,20}corticale)",
        r"(envahissement.{0,15}parties\smolles)",
        r"((l[i,y]se).{1,20}os)|ost[eé]ol[i,y]|rupture.{1,20}corticale|envahissement.{1,20}parties\smolles|ost[eé]ocondensa.{1,20}(suspect|secondaire|[ée]volutive)",
        r"(l[ée]sion|anomalie|image).{1,20}os.{1,30}(suspect|secondaire|[ée]volutive)",
        r"os.{1,30}(l[ée]sion|anomalie|image).{1,20}(suspect|secondaire|[ée]volutive)",
        r"(l[ée]sion|anomalie|image).{1,20}l[i,y]tique",
        r"(l[ée]sion|anomalie|image).{1,20}condensant.{1,20}(suspect|secondaire|[ée]volutive)",
        r"fracture.{1,30}(suspect|secondaire|[ée]volutive)",
        r"((l[ée]sion|anomalie|image|nodule).{1,80}(secondaire))",
        r"((l[ée]sion|anomalie|image|nodule)s.{1,40}suspec?ts?)",
    ],
    regex_attr="NORM",
)

default_patterns = [
    main_pattern,
    metastasis_pattern,
]
