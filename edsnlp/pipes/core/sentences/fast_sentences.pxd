from libcpp cimport bool
from libcpp.set cimport set
from spacy.tokens.doc cimport Doc
from spacy.typedefs cimport attr_t

cdef class SentenceSegmenter(object):
    cdef str name

cdef class FastSentenceSegmenter(object):
    cdef bool ignore_excluded
    cdef attr_t newline_hash
    cdef attr_t excluded_hash
    cdef attr_t endline_hash
    cdef set[attr_t] punct_chars_hash
    cdef set[attr_t] capitalized_shapes_hash
    cdef bool check_capitalized
    cdef int min_newline_count

    cdef void process(self, Doc doc) nogil
