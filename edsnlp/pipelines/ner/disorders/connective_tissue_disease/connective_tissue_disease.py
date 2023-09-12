"""`eds.connective_tissue_disease` pipeline"""
from typing import Any, Dict, List, Optional, Union

from spacy import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.base import SpanSetterArg

from ..base import DisorderMatcher
from .patterns import default_patterns


class ConnectiveTissueDiseaseMatcher(DisorderMatcher):
    """
    The `eds.connective_tissue_disease` pipeline component extracts mentions of
    connective tissue diseases.

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipelines/ner/disorders/connective_tissue_disease/patterns.py"
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
    nlp.add_pipe(f"eds.connective_tissue_disease")
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/connective-tissue-disease-examples.md"

    Parameters
    ----------
    nlp : Optional[Language]
        The pipeline object
    name : str
        The name of the component
    patterns : Optional[Dict[str, Any]]
        The patterns to use for matching
    label : str
        The label to use for the `Span` object and the extension
    span_setter : SpanSetterArg
        How to set matches on the doc

    Authors and citation
    --------------------
    The `eds.connective_tissue_disease` component was developed by AP-HP's Data Science
    team with a team of medical experts. A paper describing in details the development
    of those components is being drafted and will soon be available.
    """

    def __init__(
        self,
        nlp: Optional[Language] = None,
        name: str = "eds.connective_tissue_disease",
        *,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]] = default_patterns,
        label: str = "connective_tissue_disease",
        span_setter: SpanSetterArg = {"ents": True, "connective_tissue_disease": True},
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

            if span._.source == "lupus" and all(tok.is_upper for tok in span):
                # Huge change of FP / Title section
                continue

            yield span
