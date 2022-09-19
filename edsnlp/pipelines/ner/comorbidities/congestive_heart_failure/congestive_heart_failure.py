"""`eds.comorbidities.congestive_heart_failure` pipeline"""


from edsnlp.pipelines.ner.comorbidities.base import Comorbidity

from .patterns import default_patterns


class CongestiveHeartFailure(Comorbidity):
    def __init__(self, nlp, patterns):

        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="congestive_heart_failure",
            patterns=patterns,
        )
