mild = dict(
    source="mild",
    regex=[
        r"cholangites?.{1,10}(sclero|secondaire)",
        r"fibrose.{1,10}(hepatique|foie)",
        r"hepatite.{1,15}chroni\w+",
        r"hepatopath\w+",
        r"\bnash\b",
        r"(maladie|sydrome?).{1,10}hanot",
        r"surinfections?.{1,5}delta",
        r"\bcbp\b",
        r"\bmaf\b",
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
        r"cirr?hose",
        r"necrose.{1,10}(hepati|foie)",
        r"varice.{1,10}(estomac|oesopha|gastr)",
        r"\bvo\b.{1,5}(stade|grade).(1|2|3|i{1,3})",
        r"hypertension.{1,5}portale?",
        r"scleroses?.{1,5}hepato\s*portale?",
        r"sydrome?.{1,10}hepato.?ren",
        r"insuff?isance.{1,5}hepa",
        r"encephalopath\w+.{1,5}hepa",
        r"\btips\b",
    ],
    regex_attr="NORM",
)

transplant = dict(
    source="transplant",
    regex=[
        r"(?<!pre.?)(gref?fe|transplant).{1,12}(hepatique|foie)",
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
