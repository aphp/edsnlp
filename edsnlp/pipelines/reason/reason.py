from typing import Any, Dict, Iterable, List, Optional, Union

from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.matcher import GenericMatcher
from edsnlp.utils.filter_matches import _filter_matches
from edsnlp.utils.inclusion import check_inclusion


class Reason(GenericMatcher):
    """Pipeline to denftify the reason of the hospitalisation.

    It declares a Span extension called `reason` and adds the key ``reasons`` to doc.spans

    Parameters
    ----------
    nlp: Language
        spaCy nlp pipeline to use for matching.
    terms : Optional[Dict[str, Union[List[str], str]]]
        A dictionary of terms.
    attr: str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with the key 'term_attr'
        we can also add a key for each regex.
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
    ):
        super().__init__(
            nlp,
            terms,
            attr,
            regex,
            fuzzy=False,
            fuzzy_kwargs=None,
            filter_matches=False,
            on_ents_only=False,
        )
        self.use_sections = use_sections

        if not Span.has_extension("ents_reason"):
            Span.set_extension("ents_reason", default=None)

        if use_sections:
            self._add_section_pipeline(nlp)

    def _add_section_pipeline(self, nlp: Language):
        """Add the pipeline ``sections``

        Parameters
        ----------
        nlp: Language
            spaCy nlp pipeline to use for matching.
        """

        if not nlp.has_pipe("sections"):
            nlp.add_pipe("sections", last=True)

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
            if section.label_ in ["motif", "conclusion"]:
                reasons.append(section)

            if section.label_ in [
                "antécédents",
                "antécédents familiaux",
                "histoire de la maladie",
            ]:
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
        reasons = _filter_matches(matches, "reasons")

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

                reason._.ents_reason = ent_list

        return doc
