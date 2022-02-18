from typing import Union

import spacy

regex = [r"charlson"]

after_extract = r"charlson.*?[\n\W]*?(\d+)"

score_normalization_str = "score_normalization.charlson"


@spacy.registry.misc(score_normalization_str)
def score_normalization(extracted_score: Union[str, None]):
    """
    Charlson score normalization.
    If available, returns the integer value of the Charlson score.
    """
    score_range = list(range(0, 30))
    if (extracted_score is not None) and (int(extracted_score) in score_range):
        return int(extracted_score)
