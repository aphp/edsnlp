"""`eds.tobacco` pipeline"""

from typing import Any, Dict, List, Optional, Union

from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import SpanSetterArg
from edsnlp.utils.numbers import parse_digit

from ..alcohol.alcohol import AlcoholMatcher
from .patterns import default_patterns


class TobaccoMatcher(AlcoholMatcher):
    """
    The `eds.tobacco` pipeline component extracts mentions of tobacco consumption.

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipes/ner/behaviors/tobacco/patterns.py"
        # fmt: on
        ```

    Extensions
    ----------
    On each span `span` that match, the following attributes are available:

    - `span._.detailed_status`: either None or `"ABSTINENCE"`
    if the patient stopped its consumption
    - `span._.assigned`: dictionary with the following keys, if relevant:
        - `PA`: the mentioned *year-pack* (= *paquet-année*)
        - `secondhand`: if secondhand smoking
    - `span._.negation`: set to True when either
        - A pack-year value of 0 is extracted
        - A mention such as "tabac: 0" is found
        - The patient experiences secondhand smoking

    !!! warning "Use qualifiers !"
        Although the tobacco pipe sometime sets value for the `negation` attribute,
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
    nlp.add_pipe(eds.tobacco())
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/tobacco-examples.md"


    Parameters
    ----------
    nlp : Optional[PipelineProtocol]
        The pipeline object
    name : Optional[str]
        The name of the component
    patterns: Union[Dict[str, Any], List[Dict[str, Any]]]
        The patterns to use for matching
    label : str
        The label to use for the `Span` object and the extension
    span_setter : SpanSetterArg
        How to set matches on the doc

    Authors and citation
    --------------------
    The `eds.tobacco` component was developed by AP-HP's Data Science team with a
    team of medical experts, following the insights of the algorithm proposed
    by [@petitjean_2024].
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: str = "tobacco",
        *,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]] = default_patterns,
        label: str = "tobacco",
        span_setter: SpanSetterArg = {"ents": True, "tobacco": True},
    ):
        super().__init__(
            nlp=nlp,
            name=name,
            patterns=patterns,
            label=label,
            span_setter=span_setter,
        )

    def process(self, doc: Doc) -> List[Span]:
        for span in super().process(doc):
            if "secondhand" in span._.assigned.keys():
                span._.negation = True

            elif "PA" in span._.assigned.keys():
                pa = parse_digit(
                    span._.assigned["PA"],
                    atttr="NORM",
                    ignore_excluded=True,
                )
                if (pa == 0) and ("stopped" not in span._.assigned.keys()):
                    span._.negation = True

            yield span
