from spacy import Language
from typing import Iterable, List, Optional

from libcpp cimport bool

from spacy.attrs cimport IS_ALPHA, IS_ASCII, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_SPACE
from spacy.lexeme cimport Lexeme
from spacy.tokens.doc cimport Doc
from spacy.tokens.token cimport TokenC

from .terms import punctuation

cdef class SentenceSegmenter:
    def __init__(
          self,
          nlp: Language,
          name: Optional[str] = None,
          *,
          punct_chars: Optional[List[str]],
          use_endlines: bool,
          ignore_excluded: bool = True,
    ):
        if isinstance(nlp, Language):
            vocab = nlp.vocab
        else:
            vocab = nlp
        self.name = name

        if punct_chars is None:
            punct_chars = punctuation

        self.ignore_excluded = ignore_excluded or use_endlines
        self.newline_hash = vocab.strings["\n"]
        self.excluded_hash = vocab.strings["EXCLUDED"]
        self.endline_hash = vocab.strings["ENDLINE"]
        self.punct_chars_hash = {vocab.strings[c] for c in punct_chars}
        self.capitalized_shapes_hash = {
            vocab.strings[shape]
            for shape in ("Xx", "Xxx", "Xxxx", "Xxxxx")
        }

        if use_endlines:
            print("The use_endlines is deprecated and has been replaced by the "
                  "ignore_excluded parameter")

    def __call__(self, doc: Doc):
        self.process(doc)
        return doc

    cdef void process(self, Doc doc) nogil:
        """
        Segments the document in sentences.

        Arguments
        ---------
        docs: Iterable[Doc]
            A list of spacy Doc objects.
        """
        cdef TokenC token
        cdef bool seen_period
        cdef bool seen_newline
        cdef bool is_in_punct_chars
        cdef bool is_newline

        seen_period = False
        seen_newline = False

        if doc.length == 0:
            return

        for i in range(doc.length):
            # To set the attributes at False by default for the other tokens
            doc.c[i].sent_start = 1 if i == 0 else -1

            token = doc.c[i]

            if self.ignore_excluded and token.tag == self.excluded_hash:
                continue

            is_in_punct_chars = (
                  self.punct_chars_hash.const_find(token.lex.orth)
                  != self.punct_chars_hash.const_end()
            )
            is_newline = (
                  Lexeme.c_check_flag(token.lex, IS_SPACE)
                  and token.lex.orth == self.newline_hash
            )

            if seen_period or seen_newline:
                if seen_period and Lexeme.c_check_flag(token.lex, IS_DIGIT):
                    continue
                if (
                      is_in_punct_chars
                      or is_newline
                      or Lexeme.c_check_flag(token.lex, IS_PUNCT)
                ):
                    continue
                if seen_period:
                    doc.c[i].sent_start = 1
                    seen_newline = False
                    seen_period = False
                else:
                    doc.c[i].sent_start = (
                        1 if (
                              self.capitalized_shapes_hash.const_find(token.lex.shape)
                              != self.capitalized_shapes_hash.const_end()
                        ) else -1
                    )
                    seen_newline = False
                    seen_period = False
            elif is_in_punct_chars:
                seen_period = True
            elif is_newline:
                seen_newline = True
