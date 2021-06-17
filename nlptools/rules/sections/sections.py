from typing import List, Dict

from loguru import logger
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

from nlptools.rules.base import BaseComponent


class Sections(BaseComponent):
    """
    Divides the document into sections.
    
    By default, we are using a dataset of documents annotated for section titles,
    using the wonderful work done by Ivan Lerner.

    The component looks for section titles within the document,
    and stores them in the `section_title` extension.

    For ease-of-use, the component also populates a `section` extension,
    which contains a list of spans corresponding to the "sections" of the
    document. These span from the start of one section title to the next,
    which can introduce obvious bias should an intermediate section title
    goes undetected.
    """

    def __init__(
            self,
            nlp: Language,
            sections: Dict[str, List[str]],
    ):

        logger.warning('The component Sections is still in Beta. Use at your own risks.')

        self.nlp = nlp

        self.sections = sections

        if not Doc.has_extension('sections'):
            Doc.set_extension('sections', default=[])

        if not Doc.has_extension('section_titles'):
            Doc.set_extension('section_titles', default=[])

        if not Span.has_extension('section_title'):
            Span.set_extension('section_title', default=None)

        self.matcher = PhraseMatcher(self.nlp.vocab, attr='LOWER')
        self.build_patterns()

    def build_patterns(self) -> None:
        # efficiently build spaCy matcher patterns

        for section, expressions in self.sections.items():
            patterns = list(self.nlp.tokenizer.pipe(expressions))
            self.matcher.add(section, patterns)

    def process(self, doc: Doc) -> List[Span]:
        """
        Find section references in doc and filter out duplicates and inclusions

        Parameters
        ----------
        doc: spaCy Doc object

        Returns
        -------
        sections: List of Spans referring to sections.
        """
        matches = self.matcher(doc)

        sections = [
            Span(doc, start, end, label=self.nlp.vocab.strings[match_id])
            for match_id, start, end in matches
        ]

        sections = self._filter_matches(sections)

        return sections

    # noinspection PyProtectedMember
    def __call__(self, doc: Doc) -> Doc:
        """
        Divides the doc into sections

        Parameters
        ----------
        doc: spaCy Doc object
        
        Returns
        -------
        doc: spaCy Doc object, annotated for sections
        """
        titles = self.process(doc)

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

        doc._.sections = sections
        doc._.section_titles = titles

        return doc
