"""`eds.aids` pipeline"""
import itertools
from typing import Any, Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.base import SpanSetterArg
from edsnlp.pipelines.ner.disorders.base import DisorderMatcher
from edsnlp.pipelines.qualifiers.hypothesis import HypothesisQualifier
from edsnlp.pipelines.qualifiers.hypothesis.factory import (
    DEFAULT_CONFIG as DEFAULT_CONFIG_HYP,
)
from edsnlp.pipelines.qualifiers.negation.factory import (
    DEFAULT_CONFIG as DEFAULT_CONFIG_NEG,
)
from edsnlp.pipelines.qualifiers.negation.negation import NegationQualifier

from .patterns import default_patterns


class AIDSMatcher(DisorderMatcher):
    """
    The `eds.aids` pipeline component extracts mentions of AIDS. It will notably match:

    - Mentions of VIH/HIV at the SIDA/AIDS stage
    - Mentions of VIH/HIV with opportunistic(s) infection(s)

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipelines/ner/disorders/AIDS/patterns.py"
        # fmt: on
        ```

    !!! warning "On HIV infection"

        pre-AIDS HIV infection are not extracted, only AIDS.

    Extensions
    ----------
    On each span `span` that match, the following attributes are available:

    - `span._.detailed_status`: set to `"PRESENT"`
    - `span._.assigned`: dictionary with the following keys, if relevant:
        - `opportunist`: list of opportunist infections extracted around the HIV mention
        - `stage`: stage of the HIV infection

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
    nlp.add_pipe(f"eds.aids")
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/aids-examples.md"

    Parameters
    ----------
    nlp : Optional[Language]
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
    The `eds.aids` component was developed by AP-HP's Data Science team with a team of
    medical experts. A paper describing in details the development of those components
    is being drafted and will soon be available.
    """

    def __init__(
        self,
        nlp: Optional[Language],
        name: str = "eds.aids",
        *,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]] = default_patterns,
        label: str = "aids",
        span_setter: SpanSetterArg = {"ents": True, "aids": True},
    ):

        super().__init__(
            nlp=nlp,
            name=name,
            label=label,
            patterns=patterns,
            include_assigned=False,
            span_setter=span_setter,
        )

        self.inner_negation = NegationQualifier(
            nlp,
            **{
                **DEFAULT_CONFIG_NEG,
                "on_ents_only": "AIDS_opportunist",
            },
        )

        self.inner_hypothesis = HypothesisQualifier(
            nlp,
            **{
                **DEFAULT_CONFIG_HYP,
                "on_ents_only": "AIDS_opportunist",
            },
        )

    def process(self, doc):
        spans = list(super().process(doc))

        doc.spans["AIDS_opportunist"] = list(
            itertools.chain.from_iterable(
                [span._.assigned.get("opportunist", []) for span in spans]
            )
        )

        doc = self.inner_negation(self.inner_hypothesis(doc))

        for span in spans:
            opportunists = span._.assigned.get("opportunist", [])
            if opportunists:
                opportunists = [
                    ent
                    for ent in opportunists
                    if not (ent._.negation or ent._.hypothesis)
                ]
            stage = "stage" in span._.assigned

            if span._.source == "hiv" and not (opportunists or stage):
                continue

            yield span

        del doc.spans["AIDS_opportunist"]
