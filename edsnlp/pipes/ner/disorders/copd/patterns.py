main_pattern = dict(
    source="main",
    regex=[
        r"alveolites.{1,5}fibrosante",
        r"asthm",
        r"broncho.?pneumopathies.{1,5}chroniques.{1,5}obstru",
        r"bronchites.{1,5}chroniques.{1,5}obstru",
        r"fibro.{1,20}(poumon|pulmo|pleur)",
        r"fibrose.{1,5}interstitielle.{1,5}diffuse.{1,5}idiopathique",
        r"fibrose.{1,5}intersti",
        r"obstruction.{1,5}chronique.{1,10}voie.{1,5}aerienne",
        r"pneumoconiose",
        r"pneumo(nie|pathie).{0,15}(intersti|radiq|infiltr|fibro|organis)",
        r"poumon.{1,5}noir",
        r"sclerose.{1,5}pulmo",
        r"fibro.?elastose.{1,5}pleuro.?paren",
        r"apnee.{1,25}sommeil",
        r"emphyseme",
        r"insuffisan.{1,5}respiratoire.{1,5}chron",
        r"mucoviscidose",
        r"bronchiolite.oblilerante.{1,10}pneumo.{1,20}organis",
    ],
    regex_attr="NORM",
)

htap = dict(
    source="htap",
    regex=[
        r"\bhtap\b",
        r"hypertension.{0,10}pulmo",
        r"hypertension.{1,5}arter.{1,15}(poumon|pulmo)",
    ],
    regex_attr="NORM",
    exclude=[
        dict(
            regex="minime",
            window=(0, 3),
        ),
    ],
)

oxygen = dict(
    source="oxygen",
    regex=[
        r"oxygeno.?dependance",
        r"oxygeno.?requeran",
        r"oxygenation",
        r"oxygeno.?ther",
        r"oxygene",
    ],
    regex_attr="NORM",
    assign=[
        dict(
            name="long",
            regex=r"(long.{1,10}(?:cour|dure)|chroni|domicil)",
            window=6,
        ),
        dict(
            name="long_bis",
            regex=r"(persist|major|minor)",
            window=-6,
        ),
        dict(
            name="need",
            regex=r"(besoin)",
            window=(-6, 6),
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
        r"\bSAOS\b",
        r"\bSAS\b",
        r"\bSAHOS\b",
        r"\bBOOP\b",
    ],
    regex_attr="TEXT",
)

fid = dict(
    source="fid",
    regex=[
        r"\bfid\b",
    ],
    regex_attr="NORM",
    exclude=[
        dict(
            regex=[
                r"\bfig\b",
                r"palpation",
            ],
            window=(-7, 7),
        ),
    ],
)

default_patterns = [
    main_pattern,
    htap,
    oxygen,
    acronym,
    fid,
]
