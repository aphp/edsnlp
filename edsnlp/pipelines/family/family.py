from typing import Any, Dict, List, Optional, Union

from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.pipelines.matcher import GenericMatcher
from edsnlp.utils.inclusion import check_inclusion


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

    def __init__(
        self,
        nlp: Language,
        family: List[str],
        fuzzy: Optional[bool],
        filter_matches: Optional[bool],
        attr: str,
        explain: bool,
        on_ents_only: bool,
        regex: Optional[Dict[str, Union[List[str], str]]],
        fuzzy_kwargs: Optional[Dict[str, Any]],
        use_sections: bool = False,
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

        self.sections = use_sections and "sections" in self.nlp.pipe_names

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

        sections = []

        if self.sections:
            sections = [
                Span(doc, section.start, section.end, label="FAMILY")
                for section in doc.spans["sections"]
                if section.label_ == "antécédents familiaux"
            ]

        for start, end in boundaries:
            ents = [ent for ent in doc.ents if check_inclusion(ent, start, end)]

            if self.on_ents_only and not ents:
                continue

            cues = [m for m in matches if start <= m.start < end]
            cues += [s._.section_title for s in sections if doc[start] in s]

            family = bool(cues)

            if not family:
                continue

            if not self.on_ents_only:
                for token in doc[start:end]:
                    token._.family = True

            for ent in ents:
                ent._.family = True
                ent._.family_cues += cues
                if not self.on_ents_only:
                    for token in ent:
                        token._.family = True

        return doc
