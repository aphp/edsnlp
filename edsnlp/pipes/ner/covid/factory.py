from typing import Dict, List, Union

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipes.core.matcher.matcher import GenericMatcher

from ...base import SpanSetterArg
from .patterns import patterns

DEFAULT_CONFIG = dict(
    attr="LOWER",
    ignore_excluded=False,
    ignore_space_tokens=False,
    patterns=patterns,
    label="covid",
    span_setter={"ents": True, "covid": True},
)


@registry.factory.register(
    "eds.covid",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str = "covid",
    *,
    attr: Union[str, Dict[str, str]] = "LOWER",
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    patterns: List[str] = patterns,
    label: str = "covid",
    span_setter: SpanSetterArg = {"ents": True, "covid": True},
):
    """
    The `eds.covid` pipeline component detects mentions of COVID19.

    Examples
    --------
    ```python
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.covid())

    text = "Le patient est admis pour une infection au coronavirus."

    doc = nlp(text)

    doc.ents
    # Out: (infection au coronavirus,)
    ```

    Parameters
    ----------
    nlp : PipelineProtocol
        spaCy `Language` object.
    name : str
        The name of the pipe
    attr : Union[str, Dict[str, str]]
        Attribute to match on, eg `TEXT`, `NORM`, etc.
    ignore_excluded : bool
        Whether to skip excluded tokens during matching.
    ignore_space_tokens : bool
        Whether to skip space tokens during matching.
    patterns : List[str]
        The regex pattern to use
    label : str
        Label to use for matches
    span_setter : SpanSetterArg
        How to set matches on the doc

    Returns
    -------
    GenericMatcher

    Authors and citation
    --------------------
    The `eds.covid` pipeline was developed by AP-HP's Data Science team.
    """

    return GenericMatcher(
        nlp=nlp,
        name=name,
        terms=None,
        regex={label: patterns},
        attr=attr,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
        span_setter=span_setter,
    )
