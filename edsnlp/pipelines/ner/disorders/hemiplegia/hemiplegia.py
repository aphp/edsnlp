"""`eds.hemiplegia` pipeline"""


from edsnlp.pipelines.ner.disorders.base import DisorderMatcher

from .patterns import default_patterns


class Hemiplegia(DisorderMatcher):
    def __init__(self, nlp, name, patterns):

        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name=name,
            label_name="hemiplegia",
            patterns=patterns,
        )
