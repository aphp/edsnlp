"""`eds.lymphoma` pipeline"""
from typing import Any, Dict, List, Optional, Union

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import SpanSetterArg
from edsnlp.pipes.ner.disorders.base import DisorderMatcher

from .patterns import default_patterns


class LymphomaMatcher(DisorderMatcher):
    """
    The `eds.lymphoma` pipeline component extracts mentions of lymphoma.

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipes/ner/disorders/lymphoma/patterns.py"
        # fmt: on
        ```

    Extensions
    ----------
    On each span `span` that match, the following attributes are available:

    - `span._.detailed_status`: set to `"PRESENT"`

    !!! warning "Monoclonal gammapathy"

        Monoclonal gammapathies are not extracted by this pipeline

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
    nlp.add_pipe(eds.lymphoma())
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/lymphoma-examples.md"

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
    The `eds.lymphoma` component was developed by AP-HP's Data Science team with a team
    of medical experts. A paper describing in details the development of those
    components is being drafted and will soon be available.
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: str = "lymphoma",
        *,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]] = default_patterns,
        label: str = "lymphoma",
        span_setter: SpanSetterArg = {"ents": True, "lymphoma": True},
    ):
        super().__init__(
            nlp=nlp,
            name=name,
            patterns=patterns,
            label=label,
            span_setter=span_setter,
        )
