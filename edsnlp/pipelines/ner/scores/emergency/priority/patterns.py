from typing import Union

import spacy

regex = [r"\bpriorite\b"]

value_extract = r"^.*?[\n\W]*?(\d+)"

score_normalization_str = "score_normalization.priority"


@spacy.registry.misc(score_normalization_str)
def score_normalization(extracted_score: Union[str, None]):
    """
    Priority score normalization.
    If available, returns the integer value of the priority score.
    """
    score_range = list(range(0, 6))
    if (extracted_score is not None) and (int(extracted_score) in score_range):
        return int(extracted_score)
