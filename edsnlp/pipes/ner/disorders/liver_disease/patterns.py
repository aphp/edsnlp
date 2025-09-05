mild = dict(
    source="mild",
    regex=[
        r"cholangites?.{1,10}(sclero|secondaire)",
        r"fibrose.{1,10}(hepatique|foie)",
        r"hepatite.{1,15}chronique",
        r"hepatopathie",
        r"\bnash\b",
        r"(maladie|sydrome).{1,10}Hanot",
        r"surinfections.{1,5}delta",
        r"\bcbp\b",
        r"\bmaf\b",
        r"(maladie|syndrome).{1,8}hanot",
    ],
    regex_attr="NORM",
    exclude=dict(
        regex="\bdots?\b",
        window=-5,
    ),
)

moderate_severe = dict(
    source="moderate_severe",
    regex=[
        r"cirrhose",
        r"necrose.{1,10}(hepati|foie)",
        r"varice.{1,10}(estomac|oesopha|gastr)",
        r"\bvo\b.{1,5}(stade|grade).(1|2|3|i{1,3})",
        r"hypertension.{1,5}portale",
        r"scleroses.{1,5}hepatoportale",
        r"sydrome.{1,10}hepato.?ren",
        r"insuffisance.{1,5}hepa",
        r"encephalopathie.{1,5}hepa",
        r"\btips\b",
    ],
    regex_attr="NORM",
)

transplant = dict(
    source="transplant",
    regex=[
        r"(?<!pre.?)(greffe|transplant).{1,12}(hepatique|foie)",
    ],
    regex_attr="NORM",
    exclude=dict(
        regex="chc",
        window=(-5, 5),
    ),
)

default_patterns = [
    mild,
    moderate_severe,
    transplant,
]
