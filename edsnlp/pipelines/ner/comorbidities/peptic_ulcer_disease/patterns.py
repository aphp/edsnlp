main_pattern = dict(
    source="main",
    regex=[
        r"ulcere.{1,10}gastr",
        r"ulcere.{1,10}duoden",
        r"ulcere.{1,10}antra",
        r"ulcere.{1,10}pept",
        r"ulcere.{1,10}estomac",
        r"ulcere.{1,10}curling",
        r"ulcere.{1,10}bulb",
        r"(œ|oe)sophagites? pepti.{1,10}ulcer",
    ],
    regex_attr="NORM",
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bUGD\b",
    ],
    regex_attr="TEXT",
)

default_patterns = [
    main_pattern,
    acronym,
]
