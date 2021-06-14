from typing import List, Tuple, Dict

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

from loguru import logger

from nlptools.rules.base import BaseComponent

if not Doc.has_extension('note_id'):
    Doc.set_extension('note_id', default=None)


class Sections(BaseComponent):
    """
    Divides the document into sections.
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

        self.build_patterns()

    def build_patterns(self) -> None:
        # efficiently build spaCy matcher patterns
        self.matcher = PhraseMatcher(self.nlp.vocab, attr='LOWER')

        for section, expressions in self.sections.items():
            patterns = list(self.nlp.tokenizer.pipe(expressions))
            self.matcher.add(section, None, *patterns)

    def process(self, doc: Doc) -> Tuple[List[Span], List[Span], List[Span]]:
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
        sections = self.process(doc)

        ents = []

        for s1, s2 in zip(sections[:-1], sections[1:]):
            section = Span(doc, s1.start, s2.start, label=s1.label)
            ents.append(section)

        if sections:
            ents.append(Span(doc, sections[-1].start, len(doc), label=sections[-1].label))

        doc._.sections = ents

        return doc
