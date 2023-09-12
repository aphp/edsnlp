"""`eds.dementia` pipeline"""

from typing import Any, Dict, List, Optional, Union

from spacy import Language

from edsnlp.pipelines.base import SpanSetterArg

from ..base import DisorderMatcher
from .patterns import default_patterns


class DementiaMatcher(DisorderMatcher):
    """
    The `eds.dementia` pipeline component extracts mentions of dementia.

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipelines/ner/disorders/dementia/patterns.py"
        # fmt: on
        ```

    Extensions
    ----------
    On each span `span` that match, the following attributes are available:

    - `span._.detailed_status`: set to `"PRESENT"`

    Examples
    --------
    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe(
        "eds.normalizer",
        config=dict(
            accents=True,
            lowercase=True,
            quotes=True,
            spaces=True,
            pollution=dict(
                information=True,
                bars=True,
                biology=True,
                doctors=True,
                web=True,
                coding=True,
                footer=True,
            ),
        ),
    )
    nlp.add_pipe(f"eds.dementia")
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/dementia-examples.md"

    Parameters
    ----------
    nlp : Optional[Language]
        The pipeline
    name : Optional[str]
        The name of the component
    patterns: Union[Dict[str, Any], List[Dict[str, Any]]]
        The patterns to use for matching
    label : str
        The label to use for the `Span` object and the extension
    span_setter : SpanSetterArg
        The span setter to use

    Authors and citation
    --------------------
    The `eds.dementia` component was developed by AP-HP's Data Science team with a team
    of medical experts. A paper describing in details the development of those
    components is being drafted and will soon be available.
    """

    def __init__(
        self,
        nlp: Optional[Language] = None,
        name: str = "eds.dementia",
        *,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]] = default_patterns,
        label: str = "dementia",
        span_setter: SpanSetterArg = {"ents": True, "dementia": True},
    ):
        super().__init__(
            nlp=nlp,
            name=name,
            patterns=patterns,
            label=label,
            span_setter=span_setter,
        )
