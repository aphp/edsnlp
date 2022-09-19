"""`eds.comorbidities.lymphoma` pipeline"""
from edsnlp.pipelines.ner.comorbidities.base import Comorbidity

from .patterns import default_patterns


class Lymphoma(Comorbidity):
    def __init__(self, nlp, patterns):

        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="lymphoma",
            patterns=patterns,
        )
