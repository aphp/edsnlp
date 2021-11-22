from typing import Dict, List, Union

from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.base import BaseComponent
from edsnlp.matchers.regex import RegexMatcher


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
            spaCy Doc object, annotated for pollutions.
        """
        pollutions = self.process(doc)

        for pollution in pollutions:

            for token in pollution:
                token._.keep = False

        doc.spans["pollutions"] = pollutions

        return doc
