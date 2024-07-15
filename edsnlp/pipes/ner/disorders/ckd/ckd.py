"""`eds.ckd` pipeline"""

from typing import Any, Dict, List, Optional, Union

from loguru import logger
from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import SpanSetterArg
from edsnlp.pipes.ner.disorders.base import DisorderMatcher

from .patterns import default_patterns


class CKDMatcher(DisorderMatcher):
    """

    The `eds.CKD` pipeline component extracts mentions of CKD (Chronic Kidney Disease).
    It will notably match:

    - Mentions of various diseases (see below)
    - Kidney transplantation
    - Chronic dialysis
    - Renal failure **from stage 3 to 5**. The stage is extracted by trying 3 methods:
        - Extracting the mentioned stage directly ("*IRC stade IV*")
        - Extracting the severity directly ("*IRC terminale*")
        - Extracting the mentioned GFR (DFG in french) ("*IRC avec DFG estimé à 30
          mL/min/1,73m2)*")

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipes/ner/disorders/CKD/patterns.py"
        # fmt: on
        ```

    Extensions
    ----------
    On each span `span` that match, the following attributes are available:

    - `span._.detailed_status`: set to None
    - `span._.assigned`: dictionary with the following keys, if relevant:
        - `stage`: mentioned renal failure stage
        - `status`: mentioned renal failure severity (e.g. modérée, sévère, terminale,
          etc.)
        - `dfg`: mentioned DFG

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
    nlp.add_pipe(eds.ckd())
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/ckd-examples.md"

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
    The `eds.ckd` component was developed by AP-HP's Data Science team with a
    team of medical experts, following the insights of the algorithm proposed
    by [@petitjean_2024].
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: str = "ckd",
        *,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]] = default_patterns,
        label: str = "ckd",
        span_setter: SpanSetterArg = {"ents": True, "ckd": True},
    ):
        super().__init__(
            nlp=nlp,
            name=name,
            label=label,
            patterns=patterns,
            span_setter=span_setter,
        )

    def classify_from_dfg(self, dfg_span: Optional[Span]):
        if dfg_span is None:
            return False
        try:
            dfg_value = float(dfg_span.text.replace(",", ".").strip())
        except ValueError:
            logger.trace(f"DFG value couldn't be extracted from {dfg_span.text}")
            return False

        return dfg_value < 60  # We keep only moderate to severe CKD

    def process(self, doc: Doc):
        for span in super().process(doc):
            if span._.source == "dialysis" and "chronic" not in span._.assigned.keys():
                continue

            if span._.source == "general":
                if {"stage", "status"} & set(span._.assigned.keys()):
                    yield span
                elif self.classify_from_dfg(span._.assigned.get("dfg", None)):
                    yield span
            else:
                yield span
