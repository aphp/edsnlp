import re
from typing import List, Optional, Union

from spacy.tokens import Doc, Span

from edsnlp.core.pipeline import PipelineProtocol
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.matchers.utils import Patterns
from edsnlp.pipes.base import BaseNERComponent
from edsnlp.utils.span_getters import SpanSetterArg

from .patterns import patterns


class SuicideAttemptMatcher(BaseNERComponent):
    """
    The `eds.suicide_attempt` pipeline component detects mentions of Suicide Attempt.
    It can be used with a span qualifier
    for contextualisation of the entity (history) and to detect false positives as
    negation, hypothesis or family. We recommend to use a machine learning qualifier
    to disambiguate polysemic words, as proposed in [@bey_natural_2024].


    Every matched entity will be labelled `suicide_attempt`.

    Extensions
    ----------
    Each entity span will have the suicide attempt modality as an attribute.
    The available modalities are:

    - `suicide_attempt_unspecific`
    - `autolysis`
    - `intentional_drug_overdose`
    - `jumping_from_height`
    - `cuts`
    - `strangling`
    - `self_destructive_behavior`
    - `burn_gas_caustic`

    Examples
    --------
    ```python
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.suicide_attempt())

    text = "J'ai vu le patient à cause d'une IMV avec paracetamol"
    doc = nlp(text)
    doc.ents
    # Out: (IMV,)

    ent = doc.ents[0]
    ent._.suicide_attempt_modality
    # Out: 'intentional_drug_overdose'
    ```

    ??? info "Patterns used for the named entity recognition"
        ```{ .python .no-check }
        # fmt: off
        --8<-- "edsnlp/pipes/ner/suicide_attempt/patterns.py"
        # fmt: on
        ```


    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object.
    name : str
        The name of the pipe
    attr : Union[str, Dict[str, str]]
        Attribute to match on, eg `TEXT`, `NORM`, etc.
    ignore_excluded : bool
        Whether to skip excluded tokens during matching.
    ignore_space_tokens : bool
        Whether to skip space tokens during matching.
    patterns : List[str]
        The regex pattern to use
    span_setter : SpanSetterArg
        How to set matches on the doc
    label : str
        Label name to use for the `Span` object and the extension

    Returns
    -------
    SuicideAttemptMatcher

    Authors and citation
    --------------------
    The `eds.suicide_attempt` component was developed by AP-HP's Data Science
    team, following the insights of the algorithm proposed
    by [@bey_natural_2024].
    """

    def __init__(
        self,
        nlp: PipelineProtocol,
        label: str = "suicide_attempt",
        name: Optional[str] = "eds.suicide_attempt",
        regex: Patterns = patterns,
        alignment_mode: str = "expand",
        attr: str = "TEXT",
        ignore_excluded: bool = False,
        ignore_space_tokens: bool = False,
        flags: Union[re.RegexFlag, int] = 0,  # No additional flags
        span_from_group: bool = False,
        span_setter: SpanSetterArg = {"ents": True},
    ):
        self.attr = attr
        self.label = label
        self.regex_matcher = RegexMatcher(
            alignment_mode=alignment_mode,
            attr=attr,
            ignore_excluded=ignore_excluded,
            ignore_space_tokens=ignore_space_tokens,
            flags=flags,
            span_from_group=span_from_group,
        )

        self.regex_matcher.build_patterns(regex=regex)

        super().__init__(nlp=nlp, name=name, span_setter=span_setter)

    def set_extensions(self):
        if not Span.has_extension("suicide_attempt_modality"):
            Span.set_extension("suicide_attempt_modality", default=None)

    def process(self, doc: Doc) -> List[Span]:
        """
        Find matching spans in doc.

        Parameters
        ----------
        doc:
            Doc object.

        Returns
        -------
        spans:
            List of Spans returned by the matchers.
        """

        spans = list(self.regex_matcher(doc, as_spans=True))

        for span in spans:
            span._.suicide_attempt_modality = span.label_

            span.label_ = self.label

        return spans

    def __call__(self, doc: Doc) -> Doc:
        """
        Adds spans to document.

        Parameters
        ----------
        doc:
            Doc object

        Returns
        -------
        doc:
            Doc object, annotated with suicide attempts entities.
        """
        matches = self.process(doc)

        self.set_spans(doc, matches)

        return doc
