"""`eds.myocardial_infarction` pipeline"""
from typing import Any, Dict, List, Optional, Union

from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import SpanSetterArg
from edsnlp.pipes.ner.disorders.base import DisorderMatcher

from .patterns import default_patterns


class MyocardialInfarctionMatcher(DisorderMatcher):
    """
    The `eds.myocardial_infarction` pipeline component extracts mentions of myocardial
    infarction. It will notably match:

    - Mentions of various diseases (see below)
    - Mentions of stents with a heart localization

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipes/ner/disorders/myocardial_infarction/patterns.py"
        # fmt: on
        ```

    Extensions
    ----------
    On each span `span` that match, the following attributes are available:

    - `span._.detailed_status`: set to `"PRESENT"`
    - `span._.assigned`: dictionary with the following keys, if relevant:
        - `heart_localized`: localization of the stent or bypass

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
    nlp.add_pipe(eds.myocardial_infarction())
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/myocardial-infarction-examples.md"

    Parameters
    ----------
    nlp : Optional[PipelineProtocol]
        The pipeline
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
    The `eds.myocardial_infarction` component was developed by AP-HP's Data Science
    team with a team of medical experts. A paper describing in details the development
    of those components is being drafted and will soon be available.
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: str = "myocardial_infarction",
        *,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]] = default_patterns,
        label: str = "myocardial_infarction",
        span_setter: SpanSetterArg = {"ents": True, "myocardial_infarction": True},
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
            if (
                span._.source == "with_localization"
                and "heart_localized" not in span._.assigned
            ):
                continue

            yield span
