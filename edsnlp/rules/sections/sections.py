from typing import List, Dict, Optional

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.rules.generic import GenericMatcher
from edsnlp.utils.spacy import check_spans_inclusion


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
            add_patterns: bool = True,
            attr: str = 'NORM',
            **kwargs,
    ):

        logger.warning('The component Sections is still in Beta. Use at your own risks.')

        self.add_patterns = add_patterns
        if add_patterns:
            for k, v in sections.items():
                with_endline = ['\n' + v_ for v_ in v]
                with_v = ['\nv ' + v_ for v_ in v]
                with_hyphen = ['\n- ' + v_ for v_ in v]
                sections[k] = with_v + with_endline + with_hyphen
                sections[k] += [v_ + ' :' for v_ in sections[k]]
                sections[k] = [ent + '\n' for ent in sections[k]]

        super().__init__(nlp, terms=sections, filter_matches=True, attr=attr, **kwargs)

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
        doc:
            spaCy Doc object
        
        Returns
        -------
        doc:
            spaCy Doc object, annotated for sections
        """
        titles = self.process(doc)

        if self.add_patterns:
            # Remove preceding newline
            titles = [Span(doc, title.start + 1, title.end - 1, label=title.label_) for title in titles]

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
        
        for ent in doc.ents:
            for section in doc._.sections:
                if check_spans_inclusion(ent, section):
                    ent._.section_title = section._.section_title
                    ent._.section = section.label_
                    break

        return doc
