"""`eds.leukemia` pipeline"""
from edsnlp.pipelines.ner.disorders.base import DisorderMatcher

from .patterns import default_patterns


class Leukemia(DisorderMatcher):
    def __init__(self, nlp, name, patterns):

        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name=name,
            label_name="leukemia",
            patterns=patterns,
        )
