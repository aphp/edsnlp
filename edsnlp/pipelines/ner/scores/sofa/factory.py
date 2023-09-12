import re
from typing import Any, Callable, Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.base import SpanSetterArg
from edsnlp.utils.deprecation import deprecated_factory

from .patterns import regex, score_normalization_str, value_extract
from .sofa import SofaMatcher

DEFAULT_CONFIG = dict(
    regex=regex,
    value_extract=value_extract,
    score_normalization=score_normalization_str,
    attr="NORM",
    window=10,
    ignore_excluded=False,
    ignore_space_tokens=False,
    flags=0,
    label="sofa",
    span_setter={"ents": True, "sofa": True},
)


@deprecated_factory(
    "SOFA",
    "eds.SOFA",
    assigns=["doc.ents", "doc.spans"],
)
@deprecated_factory(
    "eds.SOFA",
    "eds.sofa",
    assigns=["doc.ents", "doc.spans"],
)
@Language.factory(
    "eds.sofa",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: Optional[str] = None,
    *,
    regex: List[str] = regex,
    value_extract: List[Dict[str, str]] = value_extract,
    score_normalization: Union[
        str, Callable[[Union[str, None]], Any]
    ] = score_normalization_str,
    attr: str = "NORM",
    window: int = 10,
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    flags: Union[re.RegexFlag, int] = 0,
    label: str = "sofa",
    span_setter: SpanSetterArg = {"ents": True, "sofa": True},
):
    '''
    The `eds.sofa` component extracts
    [Sequential Organ Failure Assessment (SOFA) scores](\
    https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score),
    used to track a person's status during the stay in an intensive care unit to
    determine the extent of a person's organ function or rate failure.

    Examples
    --------

    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.sofa")

    text = """
    SOFA (à 24H) : 12.
    OMS:
    """

    doc = nlp(text)
    doc.ents
    # Out: (SOFA (à 24H) : 12,)
    ```

    Extensions
    ----------
    Each extraction exposes 3 extensions:

    ```python
    ent = doc.ents[0]

    ent._.score_name
    # Out: 'sofa'

    ent._.score_value
    # Out: 12

    ent._.score_method
    # Out: '24H'
    ```

    Score method can here be "24H", "Maximum", "A l'admission" or "Non précisée"

    Parameters
    ----------
    nlp : Language
        The pipeline object
    name : Optional[str]
        The name of the component
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
    label: str
        Label name to use for the `Span` object and the extension
    span_setter: SpanSetterArg
        How to set matches on the doc
    '''
    return SofaMatcher(
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
