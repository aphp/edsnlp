from typing import Dict, Iterable, List, Optional, Union

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.core.matcher import GenericMatcher
from edsnlp.pipelines.misc.reason import patterns
from edsnlp.utils.filter import get_spans
from edsnlp.utils.inclusion import check_inclusion


class Reason(GenericMatcher):
    """Pipeline to identify the reason of the hospitalisation.

    It declares a Span extension called `ents_reason` and adds
    the key `reasons` to doc.spans.

    It also declares the boolean extension `is_reason`.
    This extension is set to True for the Reason Spans but also
    for the entities that overlap the reason span.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    reasons : Optional[Dict[str, Union[List[str], str]]]
        The terminology of reasons.
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with
        the key 'term_attr'. We can also add a key for each regex.
    use_sections : bool,
        whether or not use the `sections` pipeline to improve results.
    ignore_excluded : bool
        Whether to skip excluded tokens.
    """

    def __init__(
        self,
        nlp: Language,
        reasons: Optional[Dict[str, Union[List[str], str]]],
        attr: Union[Dict[str, str], str],
        use_sections: bool,
        ignore_excluded: bool,
    ):

        if reasons is None:
            reasons = patterns.reasons

        super().__init__(
            nlp,
            terms=None,
            regex=reasons,
            attr=attr,
            ignore_excluded=ignore_excluded,
        )

        self.use_sections = use_sections and (
            "eds.sections" in self.nlp.pipe_names or "sections" in self.nlp.pipe_names
        )
        if use_sections and not self.use_sections:
            logger.warning(
                "You have requested that the pipeline use annotations "
                "provided by the `eds.section` pipeline, but it was not set. "
                "Skipping that step."
            )

        self.set_extensions()

    @classmethod
    def set_extensions(cls) -> None:

        if not Span.has_extension("ents_reason"):
            Span.set_extension("ents_reason", default=None)

        if not Span.has_extension("is_reason"):
            Span.set_extension("is_reason", default=False)

    def _enhance_with_sections(self, sections: Iterable, reasons: Iterable) -> List:
        """Enhance the list of reasons with the section information.
        If the reason overlaps with history, so it will be removed from the list

        Parameters
        ----------
        sections : Iterable
            Spans of sections identified with the `sections` pipeline
        reasons : Iterable
            Reasons list identified by the regex

        Returns
        -------
        List
            Updated list of spans reasons
        """

        for section in sections:
            if section.label_ in patterns.sections_reason:
                reasons.append(section)

            if section.label_ in patterns.section_exclude:
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
