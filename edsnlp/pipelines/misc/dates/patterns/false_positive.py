from edsnlp.utils.regex import make_pattern

from .atomic.delimiters import delimiters

# Pagination
page_patterns = [r"\d\/\d"]

# Phone numbers
phone_patterns = [r"(\d\d" + delimiter + r"){3,}\d\d" for delimiter in delimiters]

false_positive_pattern = make_pattern(page_patterns + phone_patterns)
