import re
from typing import Union

import spacy

regex = [r"[Ee]lston (& |et |and )?[Ee]llis", r"\b[Ee]{2}\b"]

pattern1 = r"[^\d\(\)]*[0-3]"
pattern2 = r".{0,2}[\+,]"
value_extract = rf"(?s).(\({pattern1}{pattern2}{pattern1}{pattern2}{pattern1}\))"

score_normalization_str = "score_normalization.elstonellis"


@spacy.registry.misc(score_normalization_str)
def score_normalization(extracted_score: Union[str, None]):
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
