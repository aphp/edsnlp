from typing import Dict, List

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from edsnlp.pipelines.core.matcher import GenericMatcher

from . import patterns


class Sections(GenericMatcher):
    """
    Divides the document into sections.

    By default, we are using a dataset of documents annotated for section titles,
    using the work done by Ivan Lerner, reviewed by Gilles Chatellier.

    Detected sections are :

    - allergies ;
    - antécédents ;
    - antécédents familiaux ;
    - traitements entrée ;
    - conclusion ;
    - conclusion entrée ;
    - habitus ;
    - correspondants ;
    - diagnostic ;
    - données biométriques entrée ;
    - examens ;
    - examens complémentaires ;
    - facteurs de risques ;
    - histoire de la maladie ;
    - actes ;
    - motif ;
    - prescriptions ;
    - traitements sortie.

    The component looks for section titles within the document,
    and stores them in the `section_title` extension.

    For ease-of-use, the component also populates a `section` extension,
    which contains a list of spans corresponding to the "sections" of the
    document. These span from the start of one section title to the next,
    which can introduce obvious bias should an intermediate section title
    goes undetected.

    Parameters
    ----------
    nlp : Language
        spaCy pipeline object.
    sections : Dict[str, List[str]]
        Dictionary of terms to look for.
    attr : str
        Default attribute to match on.
    ignore_excluded : bool
        Whether to skip excluded tokens.
    """

    def __init__(
        self,
        nlp: Language,
        sections: Dict[str, List[str]],
        add_patterns: bool,
        attr: str,
        ignore_excluded: bool,
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
            terms=None,
            regex=sections,
            attr=attr,
            ignore_excluded=ignore_excluded,
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
