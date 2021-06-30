from typing import List, Dict, Optional

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span

from nlptools.rules.generic import GenericMatcher


class Sections(GenericMatcher):
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

    Parameters
    ----------
    nlp:
        Spacy pipeline object.
    sections:
        Dictionary of terms to look for
    add_newline:
        Whether to add a new line character before each expression, to improve precision.

    Other Parameters
    ----------------
    fuzzy:
        Whether to use fuzzy matching. Be aware, this significantly increases compute time.
    """

    def __init__(
            self,
            nlp: Language,
            sections: Dict[str, List[str]],
            add_newline: Optional[bool] = False,
            **kwargs,
    ):

        logger.warning('The component Sections is still in Beta. Use at your own risks.')

        self.add_newline = add_newline
        if add_newline:
            for k, v in sections.items():
                sections[k] = ['\n' + v_ for v_ in v]

        super().__init__(nlp, terms=sections, filter_matches=True, **kwargs)

        if not Doc.has_extension('sections'):
            Doc.set_extension('sections', default=[])

        if not Doc.has_extension('section_titles'):
            Doc.set_extension('section_titles', default=[])

        if not Span.has_extension('section_title'):
            Span.set_extension('section_title', default=None)

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

        if self.add_newline:
            # Remove preceding newline
            titles = [Span(doc, title.start + 1, title.end, label=title.label_) for title in titles]

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
