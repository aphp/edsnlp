import re
import string
from typing import Callable, Optional

from spacy.language import Language
from spacy.tokens import Doc

DEFAULT_CONFIG = dict(
    scorer={"@scorers": "eds.nested_ner_scorer.v1"},
)


@Language.factory("clean-entities", default_config=DEFAULT_CONFIG)
class CleanEntities:
    def __init__(
        self,
        nlp: Language,
        name: str,
        scorer: Optional[Callable],
    ):
        """
        Removes empty entities from the document and clean entity boundaries
        """
        self.scorer = scorer

    def score(self, examples, **kwargs):
        return self.scorer(examples, **kwargs)

    def __call__(self, doc: Doc) -> Doc:
        new_ents = []
        for ent in doc.ents:
            if len(ent.text.strip(string.punctuation)) == 0:
                continue
            m = re.match(r"^\s*(.*?)\s*$", ent.text, flags=re.DOTALL)
            new_begin = m.start(1)
            new_end = m.end(1)
            new_ent = doc.char_span(
                ent[0].idx + new_begin,
                ent[0].idx + new_end,
                label=ent.label_,
                alignment_mode="expand",
            )
            if new_ent is not None:
                new_ents.append(new_ent)

        doc.ents = new_ents
        return doc
