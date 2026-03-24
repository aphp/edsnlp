import spacy
from typing import Iterable, List, Optional
import builtins

from libcpp cimport bool as cbool

from spacy.attrs cimport IS_ALPHA, IS_ASCII, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_SPACE
from spacy.lexeme cimport Lexeme
from spacy.tokens.doc cimport Doc
from spacy.tokens.token cimport TokenC
from .terms import punctuation


cdef class FastSentenceSegmenter(object):
    """
    Fast implementation of sentence segmenter.
    """
    def __init__(
            self,
            vocab,
            *,
            punct_chars,
            use_endlines = None,
            ignore_excluded = True,
            check_capitalized = True,
            capitalized_shapes = None,
            min_newline_count = 1,
            hard_newline_count = -1,
            use_bullet_start = False,
            bullet_starters = (),
    ):
        if punct_chars is None:
            punct_chars = punctuation

        if use_endlines is not None:
            print(
                "The use_endlines parameter of eds.sentences is deprecated and has been "
                "replaced by the ignore_excluded parameter"
            )

        self.ignore_excluded = builtins.bool(ignore_excluded or use_endlines)
        self.newline_hash = vocab.strings["\n"]
        self.excluded_hash = vocab.strings["EXCLUDED"]
        self.endline_hash = vocab.strings["ENDLINE"]
        self.punct_chars_hash = {vocab.strings[c] for c in punct_chars}
        self.check_capitalized = check_capitalized
        self.min_newline_count = min_newline_count
        self.hard_newline_count = hard_newline_count

        if self.check_capitalized and capitalized_shapes is not None:
            shapes = tuple(capitalized_shapes)
        else:
            shapes = ()
        self.capitalized_shapes_hash = {vocab.strings[shape] for shape in shapes}

        self.use_bullet_start = use_bullet_start
        self.bullet_starter_hash = {vocab.strings[c] for c in bullet_starters}

    def __call__(self, doc: spacy.tokens.Doc):
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
        cdef cbool seen_period
        cdef cbool is_in_punct_chars
        cdef cbool is_newline
        cdef cbool hard_newline_reached
        cdef int newline_count

        seen_period = False
        newline_count = 0

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
            is_newline = (token.lex.orth == self.newline_hash)
            if not is_newline:
                with gil:
                    ctext = (<object>doc)[i].text
                    is_newline = ('\n' in ctext) or ('\r' in ctext)

            hard_newline_reached = (
                self.hard_newline_count >= 0
                and newline_count >= self.hard_newline_count
            )
            if seen_period or newline_count >= self.min_newline_count:
                if seen_period and Lexeme.c_check_flag(token.lex, IS_DIGIT):
                    continue
                if not (
                        is_in_punct_chars
                        or is_newline
                        or (
                            Lexeme.c_check_flag(token.lex, IS_PUNCT) and (
                                self.bullet_starter_hash.const_find(token.lex.orth)
                                == self.bullet_starter_hash.const_end()
                            )
                        )
                ):
                    if seen_period:
                        doc.c[i].sent_start = 1
                        newline_count = 0
                        seen_period = False
                    else:
                        doc.c[i].sent_start = (
                            1 if (
                                hard_newline_reached or
                                not self.check_capitalized or (
                                    self.capitalized_shapes_hash.const_find(token.lex.shape)
                                    != self.capitalized_shapes_hash.const_end()
                                ) or (
                                    self.use_bullet_start and (
                                        self.bullet_starter_hash.const_find(token.lex.orth)
                                        != self.bullet_starter_hash.const_end()
                                    )
                                )
                            ) else -1
                        )
                        newline_count = 0
                        seen_period = False
                    continue
            if is_in_punct_chars:
                seen_period = True
            if is_newline:
                newline_count += 1
            else:
                newline_count = 0
