import re
from typing import Any, Callable, List, Union

from spacy.language import Language

from edsnlp.pipelines.ner.scores import Score
from edsnlp.pipelines.ner.scores.elstonellis import patterns

DEFAULT_CONFIG = dict(
    regex=patterns.regex,
    value_extract=patterns.value_extract,
    score_normalization=patterns.score_normalization_str,
    attr="TEXT",
    window=20,
    ignore_excluded=False,
    ignore_space_tokens=False,
    flags=0,
)


@Language.factory(
    "eds.elston-ellis",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    regex: List[str] = patterns.regex,
    value_extract: str = patterns.value_extract,
    score_normalization: Union[
        str, Callable[[Union[str, None]], Any]
    ] = patterns.score_normalization_str,
    attr: str = "TEXT",
    window: int = 20,
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    flags: Union[re.RegexFlag, int] = 0,
):
    """
    Matcher for the Elston-Ellis score.

    Parameters
    ----------
    nlp: Language
        The spaCy Language object
    name: str
        The name of the component
    regex: List[str]
        The regex patterns to match
    value_extract: str
        The regex pattern to extract the value from the matched text
    score_normalization: Union[str, Callable[[Union[str, None]], Any]]
        The normalization function to apply to the extracted value
    attr: str
        The token attribute to match on (e.g. "TEXT" or "NORM")
    window: int
        The window size to search for the regex pattern
    ignore_excluded: bool
        Whether to ignore excluded tokens
    ignore_space_tokens: bool
        Whether to ignore space tokens
    flags: Union[re.RegexFlag, int]
        The regex flags to use

    Returns
    -------
    Score
    """
    return Score(
        nlp,
        name=name,
        score_name="elston-ellis",
        regex=regex,
        value_extract=value_extract,
        score_normalization=score_normalization,
        attr=attr,
        window=window,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
        flags=flags,
    )
