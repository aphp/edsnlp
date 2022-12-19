from libcpp cimport bool
from libcpp.set cimport set
from libcpp.vector cimport vector
from spacy.tokens.doc cimport Doc
from spacy.typedefs cimport attr_t


cdef enum split_options: WITH_CAPITALIZED, WITH_UPPERCASE, NONE

cdef class SentenceSegmenter(object):
    cdef bool ignore_excluded
    cdef attr_t newline_hash
    cdef attr_t excluded_hash
    cdef attr_t endline_hash
    cdef set[attr_t] punct_chars_hash
    cdef set[attr_t] capitalized_shapes_hash
    cdef set[attr_t] capitalized_chars_hash
    cdef split_options split_on_newlines

    cdef void process(self, Doc doc) nogil
