"""`eds.cerebrovascular_accident` pipeline"""
from typing import Any, Dict, List, Optional, Union

from spacy import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.base import SpanSetterArg

from ..base import DisorderMatcher
from .patterns import default_patterns


class CerebrovascularAccidentMatcher(DisorderMatcher):
    """
    The `eds.cerebrovascular_accident` pipeline component extracts mentions of
    cerebrovascular accident. It will notably match:

    - Mentions of AVC/AIT
    - Mentions of bleeding, hemorrhage, thrombus, ischemia, etc., localized in the brain

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipelines/ner/disorders/cerebrovascular_accident/patterns.py"
        # fmt: on
        ```

    Extensions
    ----------
    On each span `span` that match, the following attributes are available:

    - `span._.detailed_status`: set to `"PRESENT"`

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
    nlp.add_pipe(f"eds.cerebrovascular_accident")
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/cerebrovascular-accident-examples.md"

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
        How to set matches on the doc

    Authors and citation
    --------------------
    The `eds.cerebrovascular_accident` component was developed by AP-HP's Data Science
    team with a team of medical experts. A paper describing in details the development
    of those components is being drafted and will soon be available.
    """

    def __init__(
        self,
        nlp: Optional[Language],
        name: str = "eds.cerebrovascular_accident",
        *,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]] = default_patterns,
        label: str = "cerebrovascular_accident",
        span_setter: SpanSetterArg = {"ents": True, "cerebrovascular_accident": True},
    ):

        super().__init__(
            nlp=nlp,
            name=name,
            label=label,
            patterns=patterns,
            include_assigned=False,
            span_setter=span_setter,
        )

    def process(self, doc: Doc) -> List[Span]:
        for span in super().process(doc):
            if (span._.source == "with_localization") and (
                "brain_localized" not in span._.assigned.keys()
            ):
                continue

            if span._.source == "ischemia":
                if "brain" not in span._.assigned.keys():
                    continue

            yield span
