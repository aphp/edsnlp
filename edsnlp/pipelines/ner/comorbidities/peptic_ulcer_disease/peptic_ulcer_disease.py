"""`eds.comorbidities.peptic_ulcer_disease` pipeline"""


from edsnlp.pipelines.ner.comorbidities.base import Comorbidity

from .patterns import default_patterns


class PepticUlcerDisease(Comorbidity):
    def __init__(self, nlp, patterns):

        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="peptic_ulcer_disease",
            patterns=patterns,
        )
