import re
from typing import Any, Callable, List, Optional, Union

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.base import SpanSetterArg
from edsnlp.pipes.ner.scores.base_score import SimpleScoreMatcher
from edsnlp.pipes.ner.scores.charlson import patterns

DEFAULT_CONFIG = dict(
    regex=patterns.regex,
    value_extract=patterns.value_extract,
    score_normalization=patterns.score_normalization_str,
    attr="NORM",
    window=7,
    ignore_excluded=False,
    ignore_space_tokens=False,
    flags=0,
    label="charlson",
    span_setter={"ents": True, "charlson": True},
)


@registry.factory.register(
    "eds.charlson",
    assigns=["doc.ents", "doc.spans"],
    deprecated=[
        "eds.charlson",
        "charlson",
    ],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = "charlson",
    *,
    regex: List[str] = patterns.regex,
    value_extract: str = patterns.value_extract,
    score_normalization: Union[
        str, Callable[[Union[str, None]], Any]
    ] = patterns.score_normalization_str,
    attr: str = "NORM",
    window: int = 7,
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    flags: Union[re.RegexFlag, int] = 0,
    label: str = "charlson",
    span_setter: SpanSetterArg = {"ents": True, "charlson": True},
):
    '''
    The `eds.charlson` component extracts the
    [Charlson Comorbidity Index](https://www.mdcalc.com/charlson-comorbidity-index-cci).

    Examples
    --------
    ```python
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.sentences())
    nlp.add_pipe(eds.normalizer())
    nlp.add_pipe(eds.charlson())

    text = """
    Charlson à l'admission: 7.
    Charlson:
    OMS:
    """

    doc = nlp(text)
    doc.ents
    # Out: (Charlson à l'admission: 7,)
    ```

    We can see that only one occurrence was extracted. The second mention of
    Charlson in the text doesn't contain any numerical value, so it isn't extracted.

    Extensions
    ----------
    Each extraction exposes 2 extensions:

    ```python
    ent = doc.ents[0]

    ent._.score_name
    # Out: 'charlson'

    ent._.score_value
    # Out: 7
    ```

    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object
    name : Optional[str]
        Name of the component
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
    label : str
        Label name to use for the `Span` object and the extension
    span_setter: SpanSetterArg
        How to set matches on the doc

    Returns
    -------
    SimpleScoreMatcher
    '''
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
