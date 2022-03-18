from itertools import chain

from spacy.language import Language
from spacy.tokens import Doc
from edsnlp.utils.filter import filter_spans

from edsnlp.matchers.regex import RegexMatcher
from spacy.matcher import PhraseMatcher

from .patterns import patterns

from edsnlp.utils.resources import get_cities


class Pseudonymisation:
    def __init__(
        self,
        nlp: Language,
        attr: str,
    ):

        cities = get_cities()

        self.regex_matcher = RegexMatcher(attr=attr)
        self.phrase_matcher = PhraseMatcher(vocab=nlp.vocab, attr=attr)

        self.regex_matcher.build_patterns(patterns)

        cities_patterns = list(nlp.pipe(list(set(cities.name))))
        self.phrase_matcher.add(key="VILLE", docs=cities_patterns)

    @staticmethod
    def set_extensions() -> None:
        pass

    def process(self, doc: Doc) -> Doc:

        matches = chain(
            self.regex_matcher(doc, as_spans=True),
            self.phrase_matcher(doc, as_spans=True),
        )
        matches = filter_spans(matches)

        doc.ents = matches

        return doc

    def __call__(self, doc: Doc) -> Doc:
        return self.process(doc)
