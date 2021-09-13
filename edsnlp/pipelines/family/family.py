from typing import List, Optional, Union, Dict, Any

from edsnlp.pipelines.generic import GenericMatcher
from spacy.language import Language
from spacy.tokens import Token, Span, Doc


class FamilyContext(GenericMatcher):
    """
    Implements a family context detection algorithm.

    The components looks for terms indicating family references in the text.

    Parameters
    ----------
    nlp: Language
        spaCy nlp pipeline to use for matching.
    family: List[str]
        List of terms indicating family reference.
    fuzzy: bool
         Whether to perform fuzzy matching on the terms.
    filter_matches: bool
        Whether to filter out overlapping matches.
    annotation_scheme: str
        Whether to require that all tokens in the matching span possess the desired label (`annotation_scheme = 'all'`),
        or at least one token matching (`annotation_scheme = 'any'`).
    attr: str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with the key 'term_attr'
        we can also add a key for each regex.
    on_ents_only: bool
        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.
    regex: Optional[Dict[str, Union[List[str], str]]]
        A dictionnary of regex patterns.
    fuzzy_kwargs: Optional[Dict[str, Any]]
        Default options for the fuzzy matcher, if used.
    """

    split_on_punctuation = False

    def __init__(
        self,
        nlp: Language,
        family: List[str],
        fuzzy: Optional[bool],
        filter_matches: Optional[bool],
        annotation_scheme: Optional[str],
        attr: str,
        on_ents_only: bool,
        regex: Optional[Dict[str, Union[List[str], str]]],
        fuzzy_kwargs: Optional[Dict[str, Any]],
        **kwargs,
    ):

        super().__init__(
            nlp,
            terms=dict(family=family),
            fuzzy=fuzzy,
            filter_matches=filter_matches,
            attr=attr,
            on_ents_only=on_ents_only,
            regex=regex,
            fuzzy_kwargs=fuzzy_kwargs,
            **kwargs,
        )

        self.annotation_scheme = annotation_scheme

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

        if not Doc.has_extension("family"):
            Doc.set_extension("family", default=[])

        self.sections = "sections" in self.nlp.pipe_names

    def annotate_entity(self, span: Span) -> bool:
        """
        Annotates entities.

        Parameters
        ----------
        span: A given span to annotate.

        Returns
        -------
        The annotation for the entity.
        """
        if self.annotation_scheme == "all":
            return all([t._.family for t in span])
        elif self.annotation_scheme == "any":
            return any([t._.family for t in span])

    def __call__(self, doc: Doc) -> Doc:
        """
        Finds entities related to family context.

        Parameters
        ----------
        doc: spaCy Doc object

        Returns
        -------
        doc: spaCy Doc object, annotated for context
        """
        matches = self.process(doc)
        boundaries = self._boundaries(doc)

        ents = []

        if self.sections:
            ents = [
                Span(doc, section.start, section.end, label="FAMILY")
                for section in doc._.sections
                if section.label_ == "antécédents familiaux"
            ]

        for start, end in boundaries:
            if self.on_ents_only and not doc[start:end].ents:
                continue

            sub_matches = [m for m in matches if start <= m.start < end]

            if sub_matches:
                for token in doc[start:end]:
                    token._.family = True
                ents.append(Span(doc, start, end, label="FAMILY"))

        doc._.family = ents

        for ent in ents:
            for token in ent:
                token._.family = True

        for ent in doc.ents:
            if self.annotate_entity(ent):
                # The "family" extension can be set upstream (via another pipe)
                # The family pipeline won't overwrite it if so
                ent._.family = True

        return doc
