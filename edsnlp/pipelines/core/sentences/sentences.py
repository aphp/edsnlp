from typing import List, Optional

from spacy.tokens import Doc

from .terms import punctuation


class SentenceSegmenter(object):
    """
    Segments the Doc into sentences using a rule-based strategy,
    specific to AP-HP documents.

    Applies the same rule-based pipeline as spaCy's sentencizer,
    and adds a simple rule on the new lines : if a new line is followed by a
    capitalised word, then it is also an end of sentence.

    DOCS: https://spacy.io/api/sentencizer

    Arguments
    ---------
    punct_chars : Optional[List[str]]
        Punctuation characters.
    use_endlines : bool
        Whether to use endlines prediction.
    """

    def __init__(
        self,
        punct_chars: Optional[List[str]],
        use_endlines: bool,
    ):

        if punct_chars is None:
            punct_chars = punctuation

        self.punct_chars = set(punct_chars)
        self.use_endlines = use_endlines

    def __call__(self, doc: Doc) -> Doc:
        """
        Segments the document in sentences.

        Arguments
        ---------
        doc:
            A spacy Doc object.

        Returns
        -------
        doc:
            A spaCy Doc object, annotated for sentences.
        """

        if not doc:
            return doc

        doc[0].sent_start = True

        seen_period = False
        seen_newline = False

        for i, token in enumerate(doc):
            is_in_punct_chars = token.text in self.punct_chars
            is_newline = token.is_space and "\n" in token.text

            if self.use_endlines:
                end_line = getattr(token._, "end_line", None)
                is_newline = is_newline and (end_line or end_line is None)

            token.sent_start = (
                i == 0
            )  # To set the attributes at False by default for the other tokens
            if seen_period or seen_newline:
                if token.is_punct or is_in_punct_chars or is_newline:
                    continue
                if seen_period:
                    token.sent_start = True
                    seen_newline = False
                    seen_period = False
                else:
                    token.sent_start = token.shape_.startswith("Xx")
                    seen_newline = False
                    seen_period = False
            elif is_in_punct_chars:
                seen_period = True
            elif is_newline:
                seen_newline = True

        return doc
