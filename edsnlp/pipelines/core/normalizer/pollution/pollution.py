from typing import Dict, List, Optional, Union

from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans

from . import patterns


class Pollution(BaseComponent):
    """
    Tags pollution tokens.

    Populates a number of spaCy extensions :

    - `Token._.pollution` : indicates whether the token is a pollution
    - `Doc._.clean` : lists non-pollution tokens
    - `Doc._.clean_` : original text with pollutions removed.
    - `Doc._.char_clean_span` : method to create a Span using character
      indices extracted using the cleaned text.

    Parameters
    ----------
    nlp : Language
        Language pipeline object
    pollution : Dict[str, Union[str, List[str]]]
        Dictionary containing regular expressions of pollution.
    """

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Language,
        pollution: Optional[Dict[str, Union[str, List[str]]]],
    ):

        self.nlp = nlp

        if pollution is None:
            pollution = patterns.pollution

        self.pollution = pollution

        for k, v in self.pollution.items():
            if isinstance(v, str):
                self.pollution[k] = [v]

        self.regex_matcher = RegexMatcher()
        self.build_patterns()

    def build_patterns(self) -> None:
        """
        Builds the patterns for phrase matching.
        """

        # efficiently build spaCy matcher patterns
        for k, v in self.pollution.items():
            self.regex_matcher.add(k, v)

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

        pollutions = self.regex_matcher(doc, as_spans=True)
        pollutions = filter_spans(pollutions)

        return pollutions

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
                token._.excluded = True

        doc.spans["pollutions"] = pollutions

        return doc
