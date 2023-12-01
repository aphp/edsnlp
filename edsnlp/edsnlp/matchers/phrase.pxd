from libcpp.vector cimport vector
from spacy.matcher.phrasematcher cimport PhraseMatcher
from spacy.structs cimport SpanC
from spacy.tokens.doc cimport Doc
from spacy.tokens.span cimport Span
from spacy.typedefs cimport attr_t


cdef class EDSPhraseMatcher(PhraseMatcher):
    cdef attr_t space_hash
    cdef attr_t excluded_hash

    cdef void find_matches(self, Doc doc, int start_idx, int end_idx, vector[SpanC] *matches) nogil
