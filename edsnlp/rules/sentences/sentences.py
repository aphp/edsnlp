from typing import List, Optional

from spacy.tokens import Doc


class SentenceSegmenter(object):
    """
    Segments the Doc into sentences using a rule-based strategy,
    specific to AP-HP documents.

    Applies the same rule-based pipeline as Spacy's sentencizer,
    and adds a simple rule on the new lines : if a new line is followed by a
    capitalised word, then it is also an end of sentence.

    DOCS: https://spacy.io/api/sentencizer

    Arguments
    ---------
    punct_chars:
        Punctuation characters.
    """

    # Default punctuation defined for the sentencizer : https://spacy.io/api/sentencizer
    default_punct_chars = [
        '!', '.', '?', '։', '؟', '۔', '܀', '܁', '܂', '߹',
        '।', '॥', '၊', '။', '።', '፧', '፨', '᙮', '᜵', '᜶', '᠃', '᠉', '᥄',
        '᥅', '᪨', '᪩', '᪪', '᪫', '᭚', '᭛', '᭞', '᭟', '᰻', '᰼', '᱾', '᱿',
        '‼', '‽', '⁇', '⁈', '⁉', '⸮', '⸼', '꓿', '꘎', '꘏', '꛳', '꛷', '꡶',
        '꡷', '꣎', '꣏', '꤯', '꧈', '꧉', '꩝', '꩞', '꩟', '꫰', '꫱', '꯫', '﹒',
        '﹖', '﹗', '！', '．', '？', '𐩖', '𐩗', '𑁇', '𑁈', '𑂾', '𑂿', '𑃀',
        '𑃁', '𑅁', '𑅂', '𑅃', '𑇅', '𑇆', '𑇍', '𑇞', '𑇟', '𑈸', '𑈹', '𑈻', '𑈼',
        '𑊩', '𑑋', '𑑌', '𑗂', '𑗃', '𑗉', '𑗊', '𑗋', '𑗌', '𑗍', '𑗎', '𑗏', '𑗐',
        '𑗑', '𑗒', '𑗓', '𑗔', '𑗕', '𑗖', '𑗗', '𑙁', '𑙂', '𑜼', '𑜽', '𑜾', '𑩂',
        '𑩃', '𑪛', '𑪜', '𑱁', '𑱂', '𖩮', '𖩯', '𖫵', '𖬷', '𖬸', '𖭄', '𛲟', '𝪈',
        '｡', '。'
    ]

    def __init__(self, punct_chars: Optional[List[str]] = None):
        if punct_chars:
            self.punct_chars = set(punct_chars)
        else:
            self.punct_chars = set(self.default_punct_chars)

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
            A Spacy Doc object, annotated for sentences.
        """

        seen_period = False
        seen_newline = False

        if len(doc)==0:
            return doc
        doc[0].sent_start = True

        for i, token in enumerate(doc):
            is_in_punct_chars = token.text in self.punct_chars
            is_newline = token.is_space and '\n' in token.text
            token.sent_start = (i==0) # To set the attributes at False by default for the other tokens
            if seen_period or seen_newline:
                if token.is_punct or is_in_punct_chars or is_newline:
                    continue
                if seen_period:
                    token.sent_start = True
                    seen_newline = False
                    seen_period = False
                else:
                    token.sent_start = token.shape_.startswith('Xx')
                    seen_newline = False
                    seen_period = False
            elif is_in_punct_chars:
                seen_period = True
            elif is_newline:
                seen_newline = True

        return doc
