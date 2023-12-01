from typing import Union

import spacy

regex = [r"\bsofa\b"]

digits = r"[^\d]*(\d*)"

value_extract = [
    dict(
        name="method_max",
        regex=r"(max)",
        reduce_mode="keep_first",
    ),
    dict(
        name="method_24h",
        regex=r"(24h)",
        reduce_mode="keep_first",
    ),
    dict(
        name="method_adm",
        regex=r"(admission)",
        reduce_mode="keep_first",
    ),
    dict(
        name="value",
        regex=r"^.*?[\n\W]*?(\d+)(?![h0-9])",
    ),
]

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
