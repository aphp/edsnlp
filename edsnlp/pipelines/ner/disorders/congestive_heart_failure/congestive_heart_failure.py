"""`eds.congestive_heart_failure` pipeline"""
from typing import Any, Dict, List, Optional, Union

from spacy import Language

from edsnlp.pipelines.base import SpanSetterArg

from ..base import DisorderMatcher
from .patterns import default_patterns


class CongestiveHeartFailureMatcher(DisorderMatcher):
    """
    The `eds.congestive_heart_failure` pipeline component extracts mentions of
    congestive heart failure. It will notably match:

    - Mentions of various diseases (see below)
    - Heart transplantation
    - AF (Atrial Fibrillation)
    - Pacemaker

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipelines/ner/disorders/congestive_heart_failure/patterns.py"
        # fmt: on
        ```

    Usage
    -----
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
    nlp.add_pipe(f"eds.congestive_heart_failure")
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/congestive-heart-failure-examples.md"

    Parameters
    ----------
    nlp : Optional[Language]
        The pipeline object
    name : str,
        The name of the component
    patterns : Optional[Dict[str, Any]]
        The patterns to use for matching
    label : str
        The label to use for the `Span` object and the extension
    span_setter : SpanSetterArg
        How to set matches on the doc

    Authors and citation
    --------------------
    The `eds.congestive_heart_failure` component was developed by AP-HP's Data Science
    team with a team of medical experts. A paper describing in details the development
    of those components is being drafted and will soon be available.
    """

    def __init__(
        self,
        nlp: Optional[Language] = None,
        name: str = "eds.congestive_heart_failure",
        *,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]] = default_patterns,
        label: str = "congestive_heart_failure",
        span_setter: SpanSetterArg = {"ents": True, "congestive_heart_failure": True},
    ):
        super().__init__(
            nlp=nlp,
            name=name,
            label=label,
            patterns=patterns,
            span_setter=span_setter,
        )
