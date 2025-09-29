from edsnlp.utils.regex_utils import make_pattern

covid = [
    r"covid([-\s]?19)?",
    r"sars[-\s]?cov[-\s]?2",
    r"corona[-\s]?virus",
]

diseases = [r"pneumopathies?", r"infections?"]

patterns = [r"(" + make_pattern(diseases) + r"\s[àa]u?\s)?" + make_pattern(covid)]
