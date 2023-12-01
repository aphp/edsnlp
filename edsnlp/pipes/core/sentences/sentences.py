from typing import List, Optional

from spacy.tokens import Doc

from edsnlp.core import PipelineProtocol

from ...base import BaseComponent
from .fast_sentences import FastSentenceSegmenter
from .terms import punctuation


class SentenceSegmenter(BaseComponent):
    """
    Segments the Doc into sentences using a rule-based strategy,
    specific to AP-HP documents.

    Applies the same rule-based pipeline as spaCy's sentencizer,
    and adds a simple rule on the new lines : if a new line is followed by a
    capitalised word, then it is also an end of sentence.
    """

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: Optional[str] = None,
        punct_chars: Optional[List[str]] = None,
        use_endlines: bool = None,
        ignore_excluded: bool = True,
    ):
        """
        Parameters
        ----------
        nlp: PipelineProtocol
            The EDS-NLP pipeline
        name: Optional[str]
            The name of the component
        punct_chars: Optional[List[str]]
            Punctuation characters.
        use_endlines: bool
            Whether to use endlines prediction.
        ignore_excluded: bool
            Whether to ignore excluded tokens.
        """
        super().__init__(nlp, name)

        if punct_chars is None:
            punct_chars = punctuation

        self.fast_segmenter = FastSentenceSegmenter(
            vocab=nlp.vocab,
            punct_chars=punct_chars,
            use_endlines=use_endlines,
            ignore_excluded=ignore_excluded,
        )

    def __call__(self, doc: Doc):
        return self.fast_segmenter(doc)
