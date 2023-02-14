import re
from typing import Union

import spacy

regex = [r"[Ee]lston|[Ee]ll?is", r"\b[Ee]{2}\b"]

pattern1 = r"[^\d\(\)]*[0-3]"
pattern2 = r".{0,2}[\+,\-]"
score_norm = r"(?s).(?P<score_norm>I{1,3})"
score_detail = (
    rf"(?s).(?P<score_detail>\({pattern1}{pattern2}{pattern1}{pattern2}{pattern1}"
    + r".{0,10}\))"
)


value_extract = score_norm
value_extract_detail = score_detail

score_normalization_str = "score_normalization.elstonellis"


@spacy.registry.misc(score_normalization_str)
def score_normalization(extracted_score: Union[str, None]):
    """
    Elston and Ellis score normalization.
    If available, returns the integer value of the Elston and Ellis score.
    """
    try:
        if re.match(
            r"[0-9]",
            extracted_score,
        ):
            return int(extracted_score)
        else:
            return len(extracted_score)
    except ValueError:
        return None


score_normalization_str_detail = "score_normalization.elstonellisdetail"


@spacy.registry.misc(score_normalization_str_detail)
def score_normalization_detail(extracted_score: Union[str, None]):
    """
    Elston and Ellis score normalization.
    If available, returns the integer value of the Elston and Ellis score.
    """
    try:
        x = 0
        for i in re.findall(r"[0-3]", extracted_score):
            x += int(i)

        if x <= 5:
            return 1

        elif x <= 7:
            return 2

        else:
            return 3

    except ValueError:
        return None
