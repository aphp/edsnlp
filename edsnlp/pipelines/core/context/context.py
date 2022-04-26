from typing import List

from spacy.language import Language
from spacy.tokens import Doc

from edsnlp.pipelines.base import BaseComponent


class ContextAdder(BaseComponent):
    """
    Provides a generic context adder component.

    Parameters
    ----------
    nlp : Language
        The spaCy object.
    context : List[str]
        The list of extensions to add to the `Doc`
    """

    def __init__(
        self,
        nlp: Language,
        context: List[str],
    ):

        self.nlp = nlp
        self.context = context
        self.set_extensions()

    def set_extensions(self):
        for col in self.context:
            if not Doc.has_extension(col):
                Doc.set_extension(col, default=None)

    def __call__(self, doc: Doc) -> Doc:
        return doc
