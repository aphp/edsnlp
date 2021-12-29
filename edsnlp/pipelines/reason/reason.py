from typing import Dict, Iterable, List, Optional, Union

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.matcher import GenericMatcher
from edsnlp.pipelines.reason.terms import section_exclude, sections_reason
from edsnlp.utils.filter import get_spans
from edsnlp.utils.inclusion import check_inclusion


class Reason(GenericMatcher):
    """Pipeline to denftify the reason of the hospitalisation.

    It declares a Span extension called ``ents_reason`` and adds
    the key ``reasons`` to doc.spans.

    It also declares the boolean extension ``is_reason``.
    This extension is set to True for the Reason Spans but also
    for the entities that overlap the reason span.

    Parameters
    ----------
    nlp: Language
        spaCy nlp pipeline to use for matching.
    terms : Optional[Dict[str, Union[List[str], str]]]
        A dictionary of terms.
    attr: str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with
        the key 'term_attr'. We can also add a key for each regex.
    regex: Optional[Dict[str, Union[List[str], str]]]
        A dictionnary of regex patterns.
    use_sections: bool,
        whether or not use the ``sections`` pipeline to improve results.
    """

    def __init__(
        self,
        nlp: Language,
        terms: Optional[Dict[str, Union[List[str], str]]],
        attr: Union[Dict[str, str], str],
        regex: Optional[Dict[str, Union[List[str], str]]],
        use_sections: bool,
        ignore_excluded: bool,
    ):
        super().__init__(
            nlp,
            terms=terms,
            regex=regex,
            attr=attr,
            filter_matches=False,
            on_ents_only=False,
            ignore_excluded=ignore_excluded,
        )

        self.use_sections = use_sections and "sections" in self.nlp.pipe_names
        if use_sections and not self.use_sections:
            logger.warning(
                "You have requested that the pipeline use annotations "
                "provided by the `section` pipeline, but it was not set. "
                "Skipping that step."
            )

        if not Span.has_extension("ents_reason"):
            Span.set_extension("ents_reason", default=None)

        if not Span.has_extension("is_reason"):
            Span.set_extension("is_reason", default=False)

    def _enhance_with_sections(self, sections: Iterable, reasons: Iterable) -> List:
        """Enhance the list of reasons with the section information.
        If the reason overlaps with antecedents, so it will be removed from the list

        Parameters
        ----------
        sections : Iterable
            Spans of sections identified with the ``sections`` pipeline
        reasons : Iterable
            Reasons list identified by the regex

        Returns
        -------
        List
            Updated list of spans reasons
        """

        for section in sections:
            if section.label_ in sections_reason:
                reasons.append(section)

            if section.label_ in section_exclude:
                for reason in reasons:
                    if check_inclusion(reason, section.start, section.end):
                        reasons.remove(reason)

        return reasons

    def __call__(self, doc: Doc) -> Doc:
        """Find spans related to the reasons of the hospitalisation

        Parameters
        ----------
        doc : Doc

        Returns
        -------
        Doc
        """
        matches = self.process(doc)
        reasons = get_spans(matches, "reasons")

        if self.use_sections:
            sections = doc.spans["sections"]
            reasons = self._enhance_with_sections(sections=sections, reasons=reasons)

        doc.spans["reasons"] = reasons

        # Entities
        if len(doc.ents) > 0:
            for reason in reasons:  # TODO optimize this iteration
                ent_list = []
                for ent in doc.ents:
                    if check_inclusion(ent, reason.start, reason.end):
                        ent_list.append(ent)
                        ent._.is_reason = True

                reason._.ents_reason = ent_list
                reason._.is_reason = True

        return doc
