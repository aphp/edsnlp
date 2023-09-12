from typing import Dict, Iterable, List, Optional, Union

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.core.matcher.matcher import GenericMatcher
from edsnlp.pipelines.misc.reason import patterns
from edsnlp.utils.filter import get_spans
from edsnlp.utils.inclusion import check_inclusion


class ReasonMatcher(GenericMatcher):
    '''
    The `eds.reason` matcher uses a rule-based algorithm to detect spans that relate
    to the reason of the hospitalisation. It was designed at AP-HP's EDS.

    Examples
    --------
    The following snippet matches a simple terminology, and looks for spans of
    hospitalisation reasons. It is complete and can be run _as is_.

    ```python
    import spacy

    text = """COMPTE RENDU D'HOSPITALISATION du 11/07/2018 au 12/07/2018
    MOTIF D'HOSPITALISATION
    Monsieur Dupont Jean Michel, de sexe masculin, âgée de 39 ans, née le 23/11/1978,
    a été hospitalisé du 11/08/2019 au 17/08/2019 pour attaque d'asthme.

    ANTÉCÉDENTS
    Antécédents médicaux :
    Premier épisode d'asthme en mai 2018."""

    nlp = spacy.blank("eds")

    # Extraction of entities
    nlp.add_pipe(
        "eds.matcher",
        config=dict(
            terms=dict(
                respiratoire=[
                    "asthmatique",
                    "asthme",
                    "toux",
                ]
            )
        ),
    )


    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.reason", config=dict(use_sections=True))
    doc = nlp(text)

    reason = doc.spans["reasons"][0]
    reason
    # Out: hospitalisé du 11/08/2019 au 17/08/2019 pour attaque d'asthme.

    reason._.is_reason
    # Out: True

    entities = reason._.ents_reason
    entities
    # Out: [asthme]

    entities[0].label_
    # Out: 'respiratoire'

    ent = entities[0]
    ent._.is_reason
    # Out: True
    ```

    Extensions
    ----------
    The `eds.reason` pipeline adds the key `reasons` to `doc.spans` and declares
    one extension, on the `Span` objects called `ents_reason`.

    The `ents_reason` extension is a list of named entities that overlap the `Span`,
    typically entities found in upstream components like `matcher`.

    It also declares the boolean extension `is_reason`. This extension is set to True
    for the Reason Spans but also for the entities that overlap the reason span.

    Parameters
    ----------
    nlp : Language
        The pipeline object
    name : str
        Name of the component
    reasons : Dict[str, Union[List[str], str]]
        Reason patterns
    attr : str
        Default token attribute to use to build the text to match on.
    use_sections : bool,
        Whether or not use the `sections` matcher to improve results.
    ignore_excluded : bool
        Whether to skip excluded tokens.

    Authors and citation
    --------------------
    The `eds.reason` matcher was developed by AP-HP's Data Science team.

    '''

    def __init__(
        self,
        nlp: Language,
        name: str = "eds.reason",
        *,
        reasons: Optional[Dict[str, Union[List[str], str]]] = None,
        attr: Union[Dict[str, str], str] = "TEXT",
        use_sections: bool = False,
        ignore_excluded: bool = False,
    ):

        if reasons is None:
            reasons = patterns.reasons

        super().__init__(
            nlp,
            name=name,
            terms=None,
            regex=reasons,
            attr=attr,
            ignore_excluded=ignore_excluded,
            span_setter={},
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

        # TODO: remove this extension, and filter directly in span group
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
