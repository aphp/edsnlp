import re
from typing import Any, Callable, List, Union

from spacy.language import Language

from edsnlp.pipelines.ner.scores import Score
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    attr="NORM",
    window=7,
    ignore_excluded=False,
    ignore_space_tokens=False,
    flags=0,
)


@deprecated_factory(
    "score",
    "eds.score",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
@Language.factory(
    "eds.score",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str = "eds.score",
    score_name: str = None,
    regex: List[str] = None,
    value_extract: str = None,
    score_normalization: Union[str, Callable[[Union[str, None]], Any]] = None,
    attr: str = "NORM",
    window: int = 7,
    flags: Union[re.RegexFlag, int] = 0,
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
):
    """
    Parameters
    ----------
    nlp : Language
        The spaCy object.
    name : str
        The name of the component.
    score_name : str
        The name of the extracted score
    regex : List[str]
        A list of regexes to identify the score
    attr : str
        Whether to match on the text ('TEXT') or on the normalized text ('NORM')
    value_extract : str
        Regex with capturing group to get the score value
    score_normalization : Callable[[Union[str,None]], Any]
        Function that takes the "raw" value extracted from the `value_extract` regex,
        and should return:

        - None if no score could be extracted
        - The desired score value else
    window : int
        Number of token to include after the score's mention to find the
        score's value
    ignore_excluded : bool
        Whether to ignore excluded spans when matching
    ignore_space_tokens : bool
        Whether to ignore space tokens when matching
    flags : Union[re.RegexFlag, int]
        Regex flags to use when matching
    """
    return Score(
        nlp,
        name=name,
        score_name=score_name,
        regex=regex,
        value_extract=value_extract,
        score_normalization=score_normalization,
        attr=attr,
        flags=flags,
        window=window,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
    )
