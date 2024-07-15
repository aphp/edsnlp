"""`eds.hemiplegia` pipeline"""

from typing import Any, Dict, List, Optional, Union

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import SpanSetterArg

from ..base import DisorderMatcher
from .patterns import default_patterns


class HemiplegiaMatcher(DisorderMatcher):
    """
    The `eds.hemiplegia` pipeline component extracts mentions of hemiplegia.

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipes/ner/disorders/hemiplegia/patterns.py"
        # fmt: on
        ```

    Extensions
    ----------
    On each span `span` that match, the following attributes are available:

    - `span._.detailed_status`: set to None

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
    nlp.add_pipe(eds.hemiplegia())
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/hemiplegia-examples.md"

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

    # Authors and citation

    The `eds.hemiplegia` component was developed by AP-HP's Data Science team with a
    team of medical experts, following the insights of the algorithm proposed
    by [@petitjean_2024].
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: str = "hemiplegia",
        *,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]] = default_patterns,
        label: str = "hemiplegia",
        span_setter: SpanSetterArg = {"ents": True, "hemiplegia": True},
    ):
        super().__init__(
            nlp=nlp,
            name=name,
            patterns=patterns,
            label=label,
            span_setter=span_setter,
        )
