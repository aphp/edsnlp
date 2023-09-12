from typing import Dict, List

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from edsnlp.pipelines.core.matcher.matcher import GenericMatcher

from . import patterns


class SectionsMatcher(GenericMatcher):
    '''
    The `eds.sections` component extracts section titles from clinical documents.
    A "section" is then defined as the span of text between two titles.

    Here is the list of sections that are currently targeted :

    - `allergies`
    - `antécédents`
    - `antécédents familiaux`
    - `traitements entrée`
    - `conclusion`
    - `conclusion entrée`
    - `habitus`
    - `correspondants`
    - `diagnostic`
    - `données biométriques entrée`
    - `examens`
    - `examens complémentaires`
    - `facteurs de risques`
    - `histoire de la maladie`
    - `actes`
    - `motif`
    - `prescriptions`
    - `traitements sortie`
    - `evolution`
    - `modalites sortie`
    - `vaccinations`
    - `introduction`

    <!-- ![Section extraction](/resources/sections.svg){ align=right width="35%"} -->

    Remarks :

    - section `introduction` corresponds to the span of text between the header
      "COMPTE RENDU D'HOSPITALISATION" (usually denoting the beginning of the document)
      and the title of the following detected section
    - this matcher works well for hospitalization summaries (CRH), but not necessarily
      for all types of documents (in particular for emergency or scan summaries
      CR-IMAGERIE)

    !!! warning "Experimental"

        Should you rely on `eds.sections` for critical downstream tasks, make sure to
        validate the results to make sure that the component works in your case.

    Examples
    --------
    The following snippet detects section titles. It is complete and can be run _as is_.

    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.sections")

    text = """
    CRU du 10/09/2021
    Motif :
    Patient admis pour suspicion de COVID
    """

    doc = nlp(text)

    doc.spans["section_titles"]
    # Out: [Motif]
    ```

    Extensions
    ----------
    The `eds.sections` matcher adds two fields to the `doc.spans` attribute :

    1. The `section_titles` key contains the list of all section titles extracted using
       the list declared in the `terms.py` module.
    2. The `sections` key contains a list of sections, ie spans of text between two
       section titles (or the last title and the end of the document).

    If the document has entities before calling this matcher an attribute `section`
    is added to each entity.

    Parameters
    ----------
    nlp : Language
        The pipeline object.
    sections : Dict[str, List[str]]
        Dictionary of terms to look for.
    attr : str
        Default attribute to match on.
    add_patterns : bool
        Whether add update patterns to match start / end of lines
    ignore_excluded : bool
        Whether to skip excluded tokens.

    Authors and citation
    --------------------
    The `eds.sections` matcher was developed by AP-HP's Data Science team.
    '''

    def __init__(
        self,
        nlp: Language,
        name: str = "eds.sections",
        *,
        sections: Dict[str, List[str]] = patterns.sections,
        add_patterns: bool = True,
        attr: str = "NORM",
        ignore_excluded: bool = True,
    ):
        logger.warning(
            "The component Sections is still in Beta. Use at your own risks."
        )

        if sections is None:
            sections = patterns.sections
        sections = dict(sections)

        self.add_patterns = add_patterns
        if add_patterns:

            for k, v in sections.items():
                sections[k] = [
                    r"(?<=(?:\n|^)[^\n]{0,5})" + ent + r"(?=[^\n]{0,5}\n)" for ent in v
                ]

        super().__init__(
            nlp,
            name=name,
            terms=None,
            regex=sections,
            attr=attr,
            ignore_excluded=ignore_excluded,
            span_setter={},
        )

        self.set_extensions()

        if not nlp.has_pipe("normalizer") and not nlp.has_pipe("eds.normalizer"):
            logger.warning("You should add pipe `eds.normalizer`")

    @classmethod
    def set_extensions(cls):

        if not Span.has_extension("section_title"):
            Span.set_extension("section_title", default=None)

        if not Span.has_extension("section"):
            Span.set_extension("section", default=None)

    @classmethod
    def tag_ents(cls, sections: List[Span]):
        for section in sections:
            for e in section.ents:
                e._.section = section.label_

    # noinspection PyProtectedMember
    def __call__(self, doc: Doc) -> Doc:
        """
        Divides the doc into sections

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for sections
        """
        titles = filter_spans(self.process(doc))

        sections = []

        for t1, t2 in zip(titles[:-1], titles[1:]):
            section = Span(doc, t1.start, t2.start, label=t1.label)
            section._.section_title = t1
            sections.append(section)

        if titles:
            t = titles[-1]
            section = Span(doc, t.start, len(doc), label=t.label)
            section._.section_title = t
            sections.append(section)

        doc.spans["sections"] = sections
        doc.spans["section_titles"] = titles

        self.tag_ents(sections)
        return doc
