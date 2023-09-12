"""`eds.alcohol` pipeline"""
from typing import Any, Dict, List, Optional, Union

from spacy import Language
from spacy.tokens import Doc, Span

from ...disorders.base import DisorderMatcher
from .patterns import default_patterns


class AlcoholMatcher(DisorderMatcher):
    """
    The `eds.alcohol` pipeline component extracts mentions of alcohol consumption.
    It won't match occasional consumption, nor acute intoxication.

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipelines/ner/behaviors/alcohol/patterns.py"
        # fmt: on
        ```

    Extensions
    ----------
    On each span `span` that match, the following attributes are available:

    - `span._.detailed_status`: set to either
        - `"PRESENT"`
        - `"ABSTINENCE"` if the patient stopped its consumption
        - `"ABSENT"` if the patient has no alcohol dependence

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
    nlp.add_pipe(f"eds.alcohol")
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/alcohol-examples.md"

    Parameters
    ----------
    nlp : Optional[Language]
        The pipeline object
    name : Optional[str]
        The name of the component
    patterns : Union[Dict[str, Any], List[Dict[str, Any]]]
        The patterns to use for matching
    label : str
        The label to use for the `Span` object and the extension
    span_setter : SpanSetterArg
        How to set matches on the doc

    Authors and citation
    --------------------
    The `eds.alcohol` component was developed by AP-HP's Data Science team with a team
    of medical experts. A paper describing in details the development of those
    components is being drafted and will soon be available.
    """

    def __init__(
        self,
        nlp: Optional[Language] = None,
        name: str = "eds.alcohol",
        *,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]] = default_patterns,
        label="alcohol",
        span_setter={"ents": True, "alcohol": True},
    ):
        super().__init__(
            nlp=nlp,
            name=name,
            patterns=patterns,
            detailed_status_mapping={
                0: "ABSENT",
                1: "PRESENT",
                2: "ABSTINENCE",
            },
            label=label,
            span_setter=span_setter,
        )

    def process(self, doc: Doc) -> List[Span]:
        for span in super().process(doc):
            if "stopped" in span._.assigned.keys():
                span._.status = 2

            elif "zero_after" in span._.assigned.keys():
                span._.status = 0

            yield span
