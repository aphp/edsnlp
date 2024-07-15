"""`eds.diabetes` pipeline"""

from typing import Any, Dict, List, Optional, Union

from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.matchers.utils import get_text
from edsnlp.pipes.base import SpanSetterArg
from edsnlp.pipes.core.contextual_matcher.contextual_matcher import get_window

from ..base import DisorderMatcher
from .patterns import COMPLICATIONS, default_patterns


class DiabetesMatcher(DisorderMatcher):
    """
    The `eds.diabetes` pipeline component extracts mentions of diabetes.

    ??? info "Details of the used patterns"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipes/ner/disorders/diabetes/patterns.py"
        # fmt: on
        ```

    Extensions
    ----------
    On each span `span` that match, the following attributes are available:

    - `span._.detailed_status`: set to either
        - `"WITH_COMPLICATION"` if the diabetes is  complicated (e.g., via organ
           damages)
        - `"WITHOUT_COMPLICATION"` otherwise
    - `span._.assigned`: dictionary with the following keys, if relevant:
        - `type`: type of diabetes (I or II)
        - `insulin`: if the diabetes is insulin-dependent
        - `corticoid`: if the diabetes is corticoid-induced

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
    nlp.add_pipe(eds.diabetes())
    ```

    Below are a few examples:

    --8<-- "docs/assets/fragments/diabetes-examples.md"

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
        The span setter to use

    # Authors and citation

    The `eds.diabetes` component was developed by AP-HP's Data Science team with a
    team of medical experts, following the insights of the algorithm proposed
    by [@petitjean_2024].
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: str = "diabetes",
        *,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]] = default_patterns,
        label: str = "diabetes",
        span_setter: SpanSetterArg = {"ents": True, "diabetes": True},
    ):
        super().__init__(
            nlp=nlp,
            name=name,
            patterns=patterns,
            detailed_status_mapping={
                1: "WITHOUT_COMPLICATION",
                2: "WITH_COMPLICATION",
            },
            label=label,
            span_setter=span_setter,
        )

        self.complication_matcher = RegexMatcher(
            attr="NORM", ignore_excluded=True, alignment_mode="expand"
        )
        self.complication_matcher.build_patterns(
            regex=dict(far_complications=COMPLICATIONS)
        )

    def process(self, doc: Doc) -> List[Span]:
        for span in super().process(doc):
            if span._.source == "complicated":
                span._.status = 2

            elif any([k.startswith("complicated") for k in span._.assigned.keys()]):
                span._.status = 2

            elif (
                get_text(span, "NORM", ignore_excluded=True) == "db"
            ) and not span._.assigned:
                # Mostly FP
                continue

            elif self.has_far_complications(span):
                span._.status = 2

            yield span

    def has_far_complications(self, span: Span):
        """
        Handles the common case where complications are listed as bullet points,
        sometimes fairly far from the anchor.
        """
        window = (0, 50)
        context = get_window(span, window, limit_to_sentence=False)
        if next(iter(self.complication_matcher(context)), None) is not None:
            return True
        return False
