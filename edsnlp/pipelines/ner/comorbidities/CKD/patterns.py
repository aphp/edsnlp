main_pattern = dict(
    source="main",
    regex=[
        r"glomerulonephrite",
        r"(?<!pyelo)nephrite.{1,10}chronique",
        r"nephropathie",
        r"glomerulopathie",
        r"\bGNIgA",
        r"syndrome nephrotique",
        r"nephroangiosclerose",
        r"mal de bright",
        r"(maladie|syndrome).{1,7}berger",
        r"(maladie|syndrome).{1,7}bright",
        r"rachitisme renal",
        r"sydrome.{1,5}alport",
        r"good.?pasture",
        r"siadh",
        r"tubulopathie",
    ],
    exclude=dict(
        regex=[
            "aigu",
        ],
        window=4,
    ),
    regex_attr="NORM",
)

transplantation = dict(
    source="transplantation",
    regex=[
        r"transplant.{1,15}(rein|renal)",
        r"greff.{1,10}(rein|renal)",
    ],
    regex_attr="NORM",
)

dialysis = dict(
    source="dialysis",
    regex=[
        r"\beer\b",
        r"epuration extra.*renale",
        r"dialyse",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="chronic",
            regex=r"("
            + r"|".join(
                [
                    "long",
                    "chronique",
                    "peritoneal",
                    "depuis",
                    "intermitten",
                    "quotidien",
                    "hebdo",
                    "seances",
                    "reprise",
                    "poursuite",
                    "programme",
                    r"\blun",
                    r"\bmar",
                    r"\bmer",
                    r"\bjeu",
                    r"\bven",
                    r"\bsam",
                    r"\bdim",
                ]
            )
            + r")",
            window=(-5, 5),
        ),
    ],
)

general = dict(
    source="general",
    regex=[
        r"insuffisance.{1,7}\b(rein|rena)",
        r"maladies? renales?",
        r"\birc\b",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="class",
            regex=r"\b(iii|iv|v|3|4|5)",
            window=7,
        ),
        dict(
            name="status",
            regex=r"\b(moder|sever|terminal|pre.greffe|post.greffe|dialys)",
            window=7,
        ),
        dict(
            name="dfg",
            regex=r"(?:dfg|debit.{1,10}filtration.{1,5}glomerulaire).*?(\d+[\.,]?\d+)",
            window=20,
            reduce_mode="keep_first",
        ),
    ],
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bDPCA\b",
        r"\bGNMP\b",
        r"\bGEM\b",
        r"\bNCM\b",
    ],
    regex_attr="TEXT",
)

default_patterns = [
    main_pattern,
    transplantation,
    dialysis,
    general,
    acronym,
]
