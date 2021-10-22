from typing import Any, Dict, List, Optional, Union

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans

from edsnlp.pipelines.matcher import GenericMatcher
from edsnlp.utils.filter import consume_spans
from edsnlp.utils.filter_matches import _filter_matches
from edsnlp.utils.inclusion import check_inclusion


class Antecedents(GenericMatcher):
    """
    Implements an antecedents detection algorithm.

    The components looks for terms indicating antecedents in the text.

    Parameters
    ----------
    nlp: Language
        spaCy nlp pipeline to use for matching.
    antecedents: List[str]
        List of terms indicating antecedent reference.
    termination: List[str]
        List of syntagme termination terms.
    use_sections: bool
        Whether to use section pipeline to detect antecedent section.
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
        antecedents: List[str],
        termination: List[str],
        use_sections: bool,
        fuzzy: bool,
        filter_matches: bool,
        attr: str,
        explain: bool,
        on_ents_only: bool,
        regex: Optional[Dict[str, Union[List[str], str]]],
        fuzzy_kwargs: Optional[Dict[str, Any]],
        **kwargs,
    ):

        super().__init__(
            nlp,
            terms=dict(
                termination=termination,
                antecedent=antecedents,
            ),
            fuzzy=fuzzy,
            filter_matches=filter_matches,
            attr=attr,
            on_ents_only=on_ents_only,
            regex=regex,
            fuzzy_kwargs=fuzzy_kwargs,
            **kwargs,
        )

        def antecedent_getter(token_or_span: Union[Token, Span]):
            if token_or_span._.antecedent is None:
                return "NOTSET"
            elif token_or_span._.antecedent:
                return "ATCD"
            else:
                return "CURRENT"

        if not Token.has_extension("antecedent"):
            Token.set_extension("antecedent", default=None)

        if not Token.has_extension("antecedent_"):
            Token.set_extension(
                "antecedent_",
                getter=antecedent_getter,
            )

        if not Span.has_extension("antecedent"):
            Span.set_extension("antecedent", default=None)

        if not Span.has_extension("antecedent_"):
            Span.set_extension(
                "antecedent_",
                getter=antecedent_getter,
            )

        if not Span.has_extension("antecedent_cues"):
            Span.set_extension("antecedent_cues", default=[])

        if not Doc.has_extension("antecedents"):
            Doc.set_extension("antecedents", default=[])

        self.sections = use_sections and "sections" in self.nlp.pipe_names
        if use_sections and not self.sections:
            logger.warning(
                "You have requested that the pipeline use annotations "
                "provided by the `section` pipeline, but it was not set. "
                "Skipping that step."
            )

        self.explain = explain

    def __call__(self, doc: Doc) -> Doc:
        """
        Finds entities related to antecedents.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for antecedents
        """

        matches = self.process(doc)

        terminations = _filter_matches(matches, "termination")
        boundaries = self._boundaries(doc, terminations)

        # Removes duplicate matches and pseudo-expressions in one statement
        matches = filter_spans(matches)

        entities = list(doc.ents) + list(doc.spans.get("discarded", []))
        ents = None

        sections = []

        if self.sections:
            sections = [
                Span(doc, section.start, section.end, label="ATCD")
                for section in doc.spans["sections"]
                if section.label_ == "antécédents"
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

            cues = _filter_matches(sub_matches, "antecedent")
            cues += sub_sections

            antecedent = bool(cues)

            if not self.on_ents_only:
                for token in doc[start:end]:
                    token._.antecedent = antecedent

            for ent in ents:
                ent._.antecedent = ent._.antecedent or antecedent

                if self.explain:
                    ent._.antecedent_cues += cues

                if not self.on_ents_only and ent._.antecedent:
                    for token in ent:
                        token._.antecedent = True

        return doc
