main_pattern = dict(
    source="main",
    regex=[
        r"insuffisance (cortico.?)?surrenal",
        r"maladie.{1,7}addison",
        r"hypoaldosteroni",
    ],
)

default_patterns = [
    main_pattern,
]
