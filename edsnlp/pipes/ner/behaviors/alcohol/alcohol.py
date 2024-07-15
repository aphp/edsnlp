"""`eds.alcohol` pipeline"""

from typing import Any, Dict, List, Optional, Union

from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.qualifiers.negation import NegationQualifier

from ...disorders.base import DisorderMatcher
from .patterns import default_patterns


class AlcoholMatcher(DisorderMatcher):
    """
    The `eds.alcohol` pipeline component extracts mentions of alcohol consumption.
    It won't match occasional consumption, nor acute intoxication.

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipes/ner/behaviors/alcohol/patterns.py"
        # fmt: on
        ```

    Extensions
    ----------
    On each span `span` that match, the following attributes are available:

    - `span._.detailed_status`: either None or `"ABSTINENCE"`
    if the patient stopped its consumption
    - `span._.negation`: set to True when a mention such as "alcool: 0" is found

    !!! warning "Use qualifiers !"
        Although the alcohol pipe sometime sets value for the `negation` attribute,
        *generic* qualifier should still be used after the pipe.

    Examples
    --------
    ```python
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.sentences())
    nlp.add_pipe(
        eds.normalizer(
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
    nlp : Optional[PipelineProtocol]
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
    The `eds.alcohol` component was developed by AP-HP's Data Science team with a
    team of medical experts, following the insights of the algorithm proposed
    by [@petitjean_2024].
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: str = "alcohol",
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
                1: None,
                2: "ABSTINENCE",
            },
            label=label,
            span_setter=span_setter,
            include_assigned=True,
        )
        self.nlp = nlp
        self.negation = NegationQualifier(nlp)

    def process(self, doc: Doc) -> List[Span]:
        for span in super().process(doc):
            if "stopped" in span._.assigned.keys():
                # using nlp(text) so that we don't assign negation flags on
                # the original document
                stopped = self.negation.process(span)
                if not any(stopped_token.negation for stopped_token in stopped.tokens):
                    span._.status = 2

            if "zero_after" in span._.assigned.keys():
                span._.negation = True

            yield span
