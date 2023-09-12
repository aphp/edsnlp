from libcpp cimport bool
from libcpp.set cimport set
from libcpp.vector cimport vector
from spacy.tokens.doc cimport Doc
from spacy.typedefs cimport attr_t


cdef class SentenceSegmenter(object):
    cdef str name
    cdef bool ignore_excluded
    cdef attr_t newline_hash
    cdef attr_t excluded_hash
    cdef attr_t endline_hash
    cdef set[attr_t] punct_chars_hash
    cdef set[attr_t] capitalized_shapes_hash

    cdef void process(self, Doc doc) nogil
