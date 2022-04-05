from edsnlp.utils.regex import make_pattern

from .absolute import absolute_date_pattern
from .atomic.delimiters import delimiter_pattern
from .current import current_pattern
from .relative import relative_pattern

false_positive_pattern = make_pattern(
    [
        r"(\d+" + delimiter_pattern + r"){3,}\d+(?!:\d\d)\b",
        r"\d\/\d",
    ]
)
