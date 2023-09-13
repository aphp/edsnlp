from typing import List, Optional, Set, Union

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.pipelines.base import SpanGetterArg, get_spans
from edsnlp.pipelines.qualifiers.base import RuleBasedQualifier
from edsnlp.pipelines.terminations import termination as default_termination
from edsnlp.utils.filter import consume_spans, filter_spans
from edsnlp.utils.inclusion import check_inclusion

from . import patterns


class FamilyContextQualifier(RuleBasedQualifier):
    """
    The `eds.family` component uses a simple rule-based algorithm to detect spans that
    describe a family member (or family history) of the patient rather than the
    patient themself.

    Examples
    --------
    The following snippet matches a simple terminology, and checks the family context
    of the extracted entities. It is complete, and can be run _as is_.

    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    # Dummy matcher
    nlp.add_pipe(
        "eds.matcher",
        config=dict(terms=dict(douleur="douleur", osteoporose="ostéoporose")),
    )
    nlp.add_pipe("eds.family")

    text = (
        "Le patient est admis le 23 août 2021 pour une douleur au bras. "
        "Il a des antécédents familiaux d'ostéoporose"
    )

    doc = nlp(text)

    doc.ents
    # Out: (douleur, ostéoporose)

    doc.ents[0]._.family
    # Out: False

    doc.ents[1]._.family
    # Out: True
    ```

    Extensions
    ----------
    The `eds.family` component declares two extensions, on both `Span` and `Token`
    objects :

    1. The `family` attribute is a boolean, set to `True` if the component predicts
       that the span/token relates to a family member.
    2. The `family_` property is a human-readable string, computed from the `family`
       attribute. It implements a simple getter function that outputs `PATIENT` or
       `FAMILY`, depending on the value of `family`.

    Parameters
    ----------
    nlp : Language
        The pipeline object.
    name : Optional[str]
        The component name.
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with the key 'term_attr'
        we can also add a key for each regex.
    family : Optional[List[str]]
        List of terms indicating family reference.
    termination : Optional[List[str]]
        List of syntagms termination terms.
    span_getter : SpanGetterArg
        Which entities should be classified. By default, `doc.ents`
    on_ents_only : Union[bool, str, List[str], Set[str]]
        Deprecated, use `span_getter` instead.

        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.

        - If True, will look in all ents located in `doc.ents` only
        - If an iterable of string is passed, will additionally look in `doc.spans[key]`
        for each key in the iterable
    explain : bool
        Whether to keep track of cues for each entity.
    use_sections : bool, by default `False`
        Whether to use annotated sections (namely `antécédents familiaux`).

    Authors and citation
    --------------------
    The `eds.family` component was developed by AP-HP's Data Science team.
    """

    def __init__(
        self,
        nlp: Language,
        name: Optional[str] = "eds.family",
        *,
        attr: str = "NORM",
        family: Optional[List[str]] = None,
        termination: Optional[List[str]] = None,
        use_sections: bool = True,
        span_getter: SpanGetterArg = None,
        on_ents_only: Union[bool, str, List[str], Set[str]] = True,
        explain: bool = False,
    ):
        terms = dict(
            family=patterns.family if family is None else family,
            termination=default_termination if termination is None else termination,
        )

        super().__init__(
            nlp=nlp,
            name=name,
            attr=attr,
            explain=explain,
            terms=terms,
            on_ents_only=on_ents_only,
            span_getter=span_getter,
        )

        self.sections = use_sections and (
            "eds.sections" in nlp.pipe_names or "sections" in nlp.pipe_names
        )
        if use_sections and not self.sections:
            logger.warning(
                "You have requested that the pipeline use annotations "
                "provided by the `section` pipeline, but it was not set. "
                "Skipping that step."
            )

    def set_extensions(self) -> None:
        super().set_extensions()
        for cls in (Token, Span):
            if not cls.has_extension("family"):
                cls.set_extension("family", default=False)

            if not cls.has_extension("family_"):
                cls.set_extension(
                    "family_",
                    getter=lambda token: "FAMILY" if token._.family else "PATIENT",
                )

        if not Span.has_extension("family_cues"):
            Span.set_extension("family_cues", default=[])

        if not Doc.has_extension("family"):
            Doc.set_extension("family", default=[])

    def process(self, doc: Doc) -> Doc:
        matches = self.get_matches(doc)

        terminations = [m for m in matches if m.label_ == "termination"]
        boundaries = self._boundaries(doc, terminations)

        # Removes duplicate matches and pseudo-expressions in one statement
        matches = filter_spans(matches, label_to_remove="pseudo")

        entities = list(get_spans(doc, self.span_getter))
        ents = None

        sections = []

        if self.sections:
            sections = [
                Span(doc, section.start, section.end, label="FAMILY")
                for section in doc.spans["sections"]
                if section.label_ == "antécédents familiaux"
            ]

        for start, end in boundaries:
            ents, entities = consume_spans(
                entities,
                filter=lambda s: check_inclusion(s, start, end),
                second_chance=ents,
            )

            sub_matches, matches = consume_spans(
                matches, lambda s: start <= s.start < end
            )

            sub_sections, sections = consume_spans(sections, lambda s: doc[start] in s)

            if self.on_ents_only and not ents:
                continue

            cues = [m for m in sub_matches if m.label_ == "family"]
            cues.extend(sub_sections)

            if not cues:
                continue

            if not self.on_ents_only:
                for token in doc[start:end]:
                    token._.family = True

            for ent in ents:
                ent._.family = True
                if self.explain:
                    ent._.family_cues += cues
                if not self.on_ents_only and ent._.family:
                    for token in ent:
                        token._.family = True

        return doc
