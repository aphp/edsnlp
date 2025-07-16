main_pattern = dict(
    source="main",
    regex=[
        r"glomerulo\s*neph\w+",
        r"(?<!pyelo)nephrite.{1,10}chron\w+",
        r"glomerulo\s*path\w+",
        r"\bGNIgA",
        r"syndrome?.{1,5}nephroti\w+",
        r"nephro\s*angio\s*sclerose",
        r"mal.de.bright",
        r"(maladie|syndrome?).{1,7}berger",
        r"(maladie|syndrome?).{1,7}bright",
        r"rachitisme.{1,5}renal",
        r"sydrome?.{1,5}alport",
        r"good.?pasture",
        r"siadh",
        r"tubulopath\w+",
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
        r"greff?.{1,10}(rein|renal)",
    ],
    regex_attr="NORM",
)

acute_on_chronic = dict(
    source="acute_on_chronic",
    regex=[
        r"insuff?isan.{1,10}(rein|renal).{1,5}aig.{1,10}chron",
    ],
    regex_attr="NORM",
)

dialysis = dict(
    source="dialysis",
    regex=[
        r"\beer\b",
        r"epuration extra.*renale",
        r"dialys",
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
        r"(insuff?isan|fonction|malad).{1,10}\b(rein|rena)",
        r"\bmrc[^a-z]",
        r"\birc[^a-z]",
        r"nephropath\w+",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="stage",
            regex=r"\b(iii|iv|v|3|4|5)\b",
            window=7,
            reduce_mode="keep_first",
        ),
        dict(
            name="status",
            regex=r"\b(moder|sever|terminal|pre.?greffe|post.?greffe|dialys|pre.?terminal)",  # noqa
            window=7,
            reduce_mode="keep_first",
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
    acute_on_chronic,
]
