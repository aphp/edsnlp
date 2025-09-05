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
        r"(œ|oe)sophagites.{1,5}pepti.{1,10}ulcer",
        r"gastrite.{1,20}ulcer",
        r"antrite.{1,5}ulcer",
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

generic = dict(
    source="generic",
    regex=r"ulcere",
    regex_attr="NORM",
    assign=dict(
        name="is_peptic",
        regex=r"\b(gastr|digest)",
        window=(-20, 20),
        limit_to_sentence=False,
    ),
)

default_patterns = [
    main_pattern,
    acronym,
    generic,
]
