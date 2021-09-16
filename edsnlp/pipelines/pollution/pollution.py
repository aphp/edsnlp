from typing import List, Dict, Optional, Union

import numpy as np
from spacy.language import Language
from spacy.tokens import Token, Span, Doc

from edsnlp.base import BaseComponent
from edsnlp.pipelines.pollution import terms
from edsnlp.matchers.regex import RegexMatcher


# noinspection PyProtectedMember
def clean_getter(doc: Doc) -> List[Token]:
    """
    Gets a list of tokens with pollution removed.

    Arguments
    ---------
    doc: Spacy Doc object.

    Returns
    -------
    tokens: List of clean tokens.
    """

    tokens = []

    for token in doc:
        if not token._.pollution:
            tokens.append(token)

    return tokens


# noinspection PyProtectedMember
def clean2original(doc: Doc) -> np.ndarray:
    """
    Creates an alignment array to convert spans from the cleaned
    textual representation to the original text object.

    Arguments
    ---------
    doc: Spacy Doc object.

    Returns
    -------
    alignment: Alignment array.
    """

    lengths = np.array([len(token.text_with_ws) for token in doc])
    pollution = np.array([token._.pollution for token in doc])

    current = 0

    clean = []

    for length, p in zip(lengths, pollution):
        if not p:
            current += length
        clean.append(current)
    clean = np.array(clean)

    alignment = np.stack([lengths.cumsum(), clean])

    return alignment


# noinspection PyProtectedMember
def align(doc: Doc, index: int) -> int:
    """
    Aligns a character found in the clean text with
    its index in the original text.

    Arguments
    ---------
    doc: Spacy Doc object.
    index: Character index in the clean text.

    Returns
    -------
    index: Character index in the original text.
    """

    if index < 0:
        index = len(doc._.clean_) - index

    alignment = clean2original(doc)
    offset = alignment[0] - alignment[1]

    return index + offset[alignment[1] < index][-1]


def char_clean2original(
    doc: Doc,
    start: int,
    end: int,
    alignment_mode: Optional[str] = "strict",
) -> Span:
    """
    Returns a Spacy Span object from character span computed on the clean text.

    Arguments
    ---------
    doc: Spacy Doc object
    start: Character index of the beginning of the expression in the clean text.
    end: Character index of the end of the expression in the clean text.
    alignment_mode: Alignment mode. See https://spacy.io/api/doc#char_span.

    Returns
    -------
    span: Span in the original text.
    """

    start, end = align(doc, start), align(doc, end)
    return doc.char_span(start, end, alignment_mode=alignment_mode)


class Pollution(BaseComponent):
    """
    Tags pollution tokens.

    Populates a number of Spacy extensions :

    - `Token._.pollution` : indicates whether the token is a pollution
    - `Doc._.clean` : lists non-pollution tokens
    - `Doc._.clean_` : original text with pollutions removed.
    - `Doc._.char_clean_span` : method to create a Span using character
      indices extracted using the cleaned text.

    Parameters
    ----------
    nlp:
        Language pipeline object
    pollution:
        Dictionary containing regular expressions of pollution.
    """

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Language,
        pollution: Dict[str, Union[str, List[str]]],
    ):

        self.nlp = nlp

        self.pollution = pollution

        for k, v in self.pollution.items():
            if isinstance(v, str):
                self.pollution[k] = [v]

        if not Token.has_extension("pollution"):
            Token.set_extension("pollution", default=False)

        if not Doc.has_extension("pollutions"):
            Doc.set_extension("pollutions", default=[])

        if not Doc.has_extension("clean"):
            Doc.set_extension("clean", getter=clean_getter)

        if not Doc.has_extension("clean_"):
            Doc.set_extension(
                "clean_",
                getter=lambda doc: "".join([t.text_with_ws for t in doc._.clean]),
            )

        if not Doc.has_extension("char_clean_span"):
            Doc.set_extension("char_clean_span", method=char_clean2original)

        self.matcher = RegexMatcher()
        self.build_patterns()

    def build_patterns(self) -> None:
        """
        Builds the patterns for phrase matching.
        """

        # efficiently build spaCy matcher patterns

        for k, v in self.pollution.items():
            self.matcher.add(k, v)

    def process(self, doc: Doc) -> List[Span]:
        """
        Find pollutions in doc and clean candidate negations to remove pseudo negations

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        pollution:
            list of pollution spans
        """

        pollutions = self.matcher(doc)

        pollutions = self._filter_matches(pollutions)

        return pollutions

    # noinspection PyProtectedMember
    def __call__(self, doc: Doc) -> Doc:
        """
        Tags pollutions.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for negation
        """
        pollutions = self.process(doc)

        for pollution in pollutions:

            for token in pollution:
                token._.pollution = True

        doc._.pollutions = pollutions

        return doc
