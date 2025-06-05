main_pattern = dict(
    source="main",
    regex=[
        r"hemipleg\w+",
        r"tetrapleg\w+",
        r"quadripleg\w+",
        r"parapleg\w+",
        r"neuropath\w+.{1,25}motrice.{1,30}type\s*[5V]",
        r"charcot.?marie.?tooth",
        r"loc?ked.?in",
        r"syndrome?.{1,5}(enfermement|verrouillage)|(desafferen)",
        r"paralysie.{1,10}hemicorps",
        r"paralysie.{1,10}jambe",
        r"paralysie.{1,10}membre",
        r"paralysie.{1,10}cote",
        r"paralysie.{1,5}cerebrale.{1,5}spastique",
    ],
    regex_attr="NORM",
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bLIS\b",
        r"\bNMSH\b",
    ],
    regex_attr="TEXT",
)

default_patterns = [
    main_pattern,
    acronym,
]
