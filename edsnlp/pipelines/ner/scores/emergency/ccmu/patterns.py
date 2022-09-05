from typing import Union

import spacy

regex = [r"\bccmu\b"]

value_extract = r"^.*?[\n\W]*?(\d+)"

score_normalization_str = "score_normalization.ccmu"


@spacy.registry.misc(score_normalization_str)
def score_normalization(extracted_score: Union[str, None]):
    """
    CCMU score normalization.
    If available, returns the integer value of the CCMU score.
    """
    score_range = [1, 2, 3, 4, 5]
    if (extracted_score is not None) and (int(extracted_score) in score_range):
        return int(extracted_score)
