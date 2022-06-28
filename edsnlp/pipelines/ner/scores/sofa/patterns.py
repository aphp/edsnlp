import re
from typing import Union

import spacy

regex = [r"\bsofa\b"]

digits = r"[^\d]*(\d*)"

after_extract = [
    dict(
        name="method_max",
        regex=r"sofa.*?(?:max{digits})".format(digits=digits),
    ),
    dict(
        name="method_24h",
        regex=r"sofa.*?(?:24h{digits})".format(digits=digits),
    ),
    dict(
        name="method_adm",
        regex=r"sofa.*?(?:admission{digits})".format(digits=digits),
    ),
    dict(
        name="no_method",
        regex=r"sofa.*?{digits}".format(digits=digits),
    ),
]

score_normalization_str = "score_normalization.sofa"


@spacy.registry.misc(score_normalization_str)
def score_normalization(extracted_score: Union[str, None]):
    """
    Sofa score normalization.
    If available, returns the integer value of the SOFA score.
    """
    value_regex = r".*?[\n\W]*?(\d+)"
    digit_value = re.match(
        value_regex, extracted_score
    )  # Use match instead of search to only look at the beginning
    digit_value = None if digit_value is None else digit_value.groups()[0]
    score_range = list(range(0, 30))
    if (digit_value is not None) and (int(digit_value) in score_range):
        return int(digit_value)
