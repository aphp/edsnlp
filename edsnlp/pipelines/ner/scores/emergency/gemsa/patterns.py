from typing import Union

import spacy

regex = [r"\bgemsa\b"]

value_extract = r"^.*?[\n\W]*?(\d+)"

score_normalization_str = "score_normalization.gemsa"


@spacy.registry.misc(score_normalization_str)
def score_normalization(extracted_score: Union[str, None]):
    """
    GEMSA score normalization.
    If available, returns the integer value of the GEMSA score.
    """
    score_range = [1, 2, 3, 4, 5, 6]
    if (extracted_score is not None) and (int(extracted_score) in score_range):
        return int(extracted_score)
