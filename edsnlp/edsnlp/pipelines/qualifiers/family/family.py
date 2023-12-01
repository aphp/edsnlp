from typing import List, Optional

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.pipelines.qualifiers.base import Qualifier
from edsnlp.pipelines.terminations import termination
from edsnlp.utils.filter import consume_spans, filter_spans, get_spans
from edsnlp.utils.inclusion import check_inclusion

from .patterns import family


class FamilyContext(Qualifier):
    """
    Implements a family context detection algorithm.

    The components looks for terms indicating family references in the text.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    family : Optional[List[str]]
        List of terms indicating family reference.
    terminations : Optional[List[str]]
        List of termination terms, to separate syntagmas.
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with the key 'term_attr'
        we can also add a key for each regex.
    on_ents_only : bool
        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.
    regex : Optional[Dict[str, Union[List[str], str]]]
        A dictionnary of regex patterns.
    explain : bool
        Whether to keep track of cues for each entity.
    use_sections : bool, by default `False`
        Whether to use annotated sections (namely `antécédents familiaux`).
    """

    defaults = dict(
        family=family,
        termination=termination,
    )

    def __init__(
        self,
        nlp: Language,
        attr: str,
        family: Optional[List[str]],
        termination: Optional[List[str]],
        use_sections: bool,
        explain: bool,
        on_ents_only: bool,
    ):

        terms = self.get_defaults(
            family=family,
            termination=termination,
        )

        super().__init__(
            nlp=nlp,
            attr=attr,
            on_ents_only=on_ents_only,
            explain=explain,
            **terms,
        )

        self.set_extensions()

        self.sections = use_sections and (
            "eds.sections" in nlp.pipe_names or "sections" in nlp.pipe_names
        )
        if use_sections and not self.sections:
            logger.warning(
                "You have requested that the pipeline use annotations "
                "provided by the `section` pipeline, but it was not set. "
                "Skipping that step."
            )

    @classmethod
    def set_extensions(cls) -> None:
        if not Token.has_extension("family"):
            Token.set_extension("family", default=False)

        if not Token.has_extension("family_"):
            Token.set_extension(
                "family_",
                getter=lambda token: "FAMILY" if token._.family else "PATIENT",
            )

        if not Span.has_extension("family"):
            Span.set_extension("family", default=False)

        if not Span.has_extension("family_"):
            Span.set_extension(
                "family_",
                getter=lambda span: "FAMILY" if span._.family else "PATIENT",
            )

        if not Span.has_extension("family_cues"):
            Span.set_extension("family_cues", default=[])

        if not Doc.has_extension("family"):
            Doc.set_extension("family", default=[])

    def process(self, doc: Doc) -> Doc:
        """
        Finds entities related to family context.

        Parameters
        ----------
        doc: spaCy Doc object

        Returns
        -------
        doc: spaCy Doc object, annotated for context
        """
        matches = self.get_matches(doc)

        terminations = get_spans(matches, "termination")
        boundaries = self._boundaries(doc, terminations)

        # Removes duplicate matches and pseudo-expressions in one statement
        matches = filter_spans(matches, label_to_remove="pseudo")

        entities = list(doc.ents) + list(doc.spans.get("discarded", []))
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

            cues = get_spans(sub_matches, "family")
            cues += sub_sections

            if not cues:
                continue

            family = bool(cues)

            if not family:
                continue

            if not self.on_ents_only:
                for token in doc[start:end]:
                    token._.family = True

            for ent in ents:
                ent._.family = True
                if self.explain:
                    ent._.family_cues += cues
                if not self.on_ents_only:
                    for token in ent:
                        token._.family = True

        return doc
