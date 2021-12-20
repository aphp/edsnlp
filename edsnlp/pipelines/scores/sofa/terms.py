from typing import Union

import spacy

regex = [r"\bsofa\b"]

method_regex = (
    r"sofa.*?((?P<max>max\w*)|(?P<vqheures>24h\w*)|"
    r"(?P<admission>admission\w*))(?P<after_value>(.|\n)*)"
)

value_regex = r".*?.[\n\W]*?(\d+)[^h\d]"

score_normalization_str = "score_normalization.sofa"


@spacy.registry.misc(score_normalization_str)
def score_normalization(extracted_score: Union[str, None]):
    """
    Sofa score normalization.
    If available, returns the integer value of the SOFA score.
    """
    score_range = list(range(0, 30))
    if (extracted_score is not None) and (int(extracted_score) in score_range):
        return int(extracted_score)
