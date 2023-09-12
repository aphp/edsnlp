import re
from typing import Any, Callable, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.base import SpanSetterArg
from edsnlp.pipelines.ner.scores.base_score import SimpleScoreMatcher
from edsnlp.pipelines.ner.scores.emergency.gemsa import patterns
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    regex=patterns.regex,
    value_extract=patterns.value_extract,
    score_normalization=patterns.score_normalization_str,
    attr="NORM",
    window=20,
    ignore_excluded=False,
    ignore_space_tokens=False,
    flags=0,
    label="emergency_gemsa",
    span_setter={"ents": True, "emergency_gemsa": True},
)


@deprecated_factory(
    "emergency.gemsa",
    "eds.emergency_gemsa",
    assigns=["doc.ents", "doc.spans"],
)
@deprecated_factory(
    "eds.emergency.gemsa",
    "eds.emergency_gemsa",
    assigns=["doc.ents", "doc.spans"],
)
@Language.factory(
    "eds.emergency_gemsa",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: Optional[str] = None,
    *,
    regex: List[str] = patterns.regex,
    value_extract: str = patterns.value_extract,
    score_normalization: Union[
        str, Callable[[Union[str, None]], Any]
    ] = patterns.score_normalization_str,
    attr: str = "NORM",
    window: int = 20,
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    flags: Union[re.RegexFlag, int] = 0,
    label: str = "emergency_gemsa",
    span_setter: SpanSetterArg = {"ents": True, "emergency_gemsa": True},
):
    """
    Matcher for explicit mentions of the French
    [GEMSA emergency score](http://medicalcul.free.fr/gemsa.html).

    Examples
    --------

    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.emergency_gemsa")
    ```

    Parameters
    ----------
    nlp : Language
        The pipeline object
    name : str
        The name of the component
    regex : List[str]
        A list of regexes to identify the score
    attr : str
        Whether to match on the text ('TEXT') or on the normalized text ('NORM')
    value_extract : str
        Regex with capturing group to get the score value
    score_normalization : Union[str, Callable[[Union[str,None]], Any]]
        Function that takes the "raw" value extracted from the `value_extract`
        regex and should return:

        - None if no score could be extracted
        - The desired score value otherwise
    window : int
        Number of token to include after the score's mention to find the
        score's value
    ignore_excluded : bool
        Whether to ignore excluded spans when matching
    ignore_space_tokens : bool
        Whether to ignore space tokens when matching
    flags : Union[re.RegexFlag, int]
        Regex flags to use when matching
    label : str
        Label name to use for the `Span` object and the extension
    span_setter: SpanSetterArg
        How to set matches on the doc

    Returns
    -------
    SimpleScoreMatcher
    """
    return SimpleScoreMatcher(
        nlp,
        name=name,
        regex=regex,
        value_extract=value_extract,
        score_normalization=score_normalization,
        attr=attr,
        window=window,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
        flags=flags,
        label=label,
        span_setter=span_setter,
    )
