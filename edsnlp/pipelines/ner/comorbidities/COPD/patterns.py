main_pattern = dict(
    source="main",
    regex=[
        r"alveolites? fibrosante",
        r"asthm",
        r"broncho.?pneumopathies? chroniques? obstru",
        r"bronchites? chroniques? obstru",
        r"fibro.{1,20}(poumon|pulmo|pleur)",
        r"fibroses? interstitielles? diffuses? idiopathique",
        r"obstructions? chroniques? des voies aerienne",
        r"pneumoconiose",
        r"pneumo(nie|pathie)s?.{0,15}(intersti|radiq|infiltr|fibro)",
        r"poumons* noir",
        r"scleroses* pulmo",
        r"fibro.?elastose pleuro.?paren",
        r"apnee.{1,8}sommeil",
        r"emphysemes? (severe|etendu)",
        r"insuffisan.{1,5}respiratoire.{1,5}chron",
        r"mucoviscidose",
    ],
    regex_attr="NORM",
)

htap = dict(
    source="htap",
    regex=[
        r"\bhtap\b",
        r"hypertension.{0,10}pulmo",
    ],
    regex_attr="NORM",
)

oxygen = dict(
    source="oxygen",
    regex=[
        r"oxygeno.?dependance",
        r"oxygeno.?requeran",
        r"oxygenation",
        r"oxygeno.?ther",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="long",
            regex=r"(long.{1,10}(?:cour|dure)|chroni)",
            window=6,
        ),
        dict(
            name="long_bis",
            regex=r"(persist|major|minor)",
            window=-6,
        ),
    ],
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bBPCO\b",
        r"\bFPI\b",
        r"\bOLD\b",
        r"\bFEPP\b",
        r"\bPINS\b",
        r"\bPID\b",
        r"\bFID\b",
        r"\bSAOS\b",
        r"\bSAS\b",
        r"\bSAHOS\b",
    ],
    regex_attr="TEXT",
)

default_patterns = [
    main_pattern,
    htap,
    oxygen,
    acronym,
]
