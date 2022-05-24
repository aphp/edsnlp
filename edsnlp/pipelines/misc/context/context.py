from typing import Dict, List

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.utils.filter import filter_spans
from edsnlp.utils.regex import make_pattern


class ContextPseudonymisation:
    def __init__(
        self,
        nlp: Language,
        attr: str,
        phrase: bool,
    ):

        self.nlp = nlp
        self.attr = attr

        self.phrase = phrase

    @staticmethod
    def set_extensions() -> None:
        pass

    def create_phrase_matcher(self, context: Dict[str, List[str]]) -> PhraseMatcher:

        matcher = PhraseMatcher(vocab=self.nlp.vocab, attr=self.attr)

        for k, v in context.items():
            matcher.add(key=k, docs=list(self.nlp.tokenizer.pipe(v)))

        return matcher

    def create_regex_matcher(self, context: Dict[str, List[str]]) -> RegexMatcher:

        matcher = RegexMatcher(attr=self.attr)
        context = {k: make_pattern(v).lower() for k, v in context.items()}
        matcher.build_patterns(context)
        return matcher

    def process(self, doc: Doc) -> Doc:

        context: Dict[str, List[str]] = doc._context

        if self.phrase:
            matcher = self.create_phrase_matcher(context)
        else:
            matcher = self.create_regex_matcher(context)

        matches = matcher(doc, as_spans=True)
        matches = filter_spans(matches)

        doc.ents = matches

        return doc

    def __call__(self, doc: Doc) -> Doc:
        return self.process(doc)
