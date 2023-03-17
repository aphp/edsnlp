import re
from typing import Any, Callable, Dict, List, Union

from spacy.language import Language

from edsnlp.pipelines.ner.scores.sofa import Sofa, patterns
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    regex=patterns.regex,
    value_extract=patterns.value_extract,
    score_normalization=patterns.score_normalization_str,
    attr="NORM",
    window=10,
    ignore_excluded=False,
    ignore_space_tokens=False,
    flags=0,
)


@deprecated_factory(
    "SOFA",
    "eds.SOFA",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
@Language.factory(
    "eds.SOFA",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    regex: List[str] = patterns.regex,
    value_extract: List[Dict[str, str]] = patterns.value_extract,
    score_normalization: Union[
        str, Callable[[Union[str, None]], Any]
    ] = patterns.score_normalization_str,
    attr: str = "NORM",
    window: int = 10,
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    flags: Union[re.RegexFlag, int] = 0,
):
    """
    Matcher component to extract the SOFA score

    Parameters
    ----------
    nlp : Language
        The spaCy object.
    name : str
        The name of the extracted score
    regex : List[str]
        A list of regexes to identify the SOFA score
    attr : str
        Whether to match on the text ('TEXT') or on the normalized text ('CUSTOM_NORM')
    value_extract : Dict[str, str]
        Regex to extract the score value
    score_normalization : Callable[[Union[str,None]], Any]
        Function that takes the "raw" value extracted from the `value_extract` regex,
        and should return
        - None if no score could be extracted
        - The desired score value else
    window : int
        Number of token to include after the score's mention to find the
        score's value
    ignore_excluded : bool
        Whether to ignore excluded spans
    ignore_space_tokens : bool
        Whether to ignore space tokens
    flags : Union[re.RegexFlag, int]
        Flags to pass to the regex
    """
    return Sofa(
        nlp,
        score_name=name,
        regex=regex,
        value_extract=value_extract,
        score_normalization=score_normalization,
        attr=attr,
        window=window,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
        flags=flags,
    )
