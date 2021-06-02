from typing import List, Tuple, Dict

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

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

        self.nlp = nlp

        self.sections = sections

        if not Doc.has_extension('sections'):
            Doc.set_extension('sections', default=[])

        self.build_patterns()

    def build_patterns(self) -> None:
        # efficiently build spaCy matcher patterns
        self.matcher = PhraseMatcher(self.nlp.vocab)

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


try:
    # We use a list provided by PyMedExt : 
    # https://github.com/equipe22/pymedext_eds/blob/master/pymedext_eds/constants.py
    from pymedext_eds.constants import SECTION_DICT as SECTIONS
except ImportError:
    SECTIONS = dict()

default_config = dict(
    sections=SECTIONS,
)


@Language.factory("sections", default_config=default_config)
def create_negation_component(
        nlp: Language,
        name: str,
        sections: Dict[str, List[str]],
):
    return Sections(nlp, sections=sections)
