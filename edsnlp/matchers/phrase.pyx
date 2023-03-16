# cython: infer_types=True, profile=True
import re

from preshed.maps cimport map_clear, map_get, map_init, map_iter, map_set

from spacy import Language
from tqdm import tqdm

from preshed.maps cimport MapStruct, key_t
from spacy.matcher.phrasematcher cimport PhraseMatcher
from spacy.tokens.token cimport Token
from spacy.typedefs cimport attr_t
from spacy.vocab cimport Vocab

from edsnlp.matchers.utils import Patterns


def get_normalized_variant(doclike) -> str:
    tokens = [t.text + t.whitespace_ for t in doclike if not t._.excluded]
    variant = "".join(tokens)
    variant = variant.rstrip(" ")
    variant = re.sub(r"\s+", " ", variant)
    return variant


cdef class EDSPhraseMatcher(PhraseMatcher):
    """
    PhraseMatcher that allows to skip excluded tokens.
    Adapted from https://github.com/explosion/spaCy/blob/master/spacy/matcher/phrasematcher.pyx

    Parameters
    ----------
    vocab : Vocab
        spaCy vocabulary to match on.
    attr : str
        Default attribute to match on, by default "TEXT".
        Can be overridden in the `add` method.
        To match on a custom attribute, prepend the attribute name with `_`.
    ignore_excluded : bool, optional
        Whether to ignore excluded tokens, by default True
    ignore_space_tokens : bool, optional
        Whether to exclude tokens that have a "SPACE" tag, by default False
    """

    def __init__(self, Vocab vocab, attr="ORTH", ignore_excluded=True, ignore_space_tokens=False, validate=False):
        """Initialize the PhraseMatcher.

        vocab (Vocab): The shared vocabulary.
        attr (int / str): Token attribute to match on.
        validate (bool): Perform additional validation when patterns are added.

        DOCS: https://spacy.io/api/phrasematcher#init
        """
        super().__init__(vocab, attr, validate)

        if ignore_excluded:
            self.excluded_hash = vocab.strings['EXCLUDED']
        else:
            self.excluded_hash = 0

        if ignore_space_tokens:
            self.space_hash = vocab.strings['SPACE']
        else:
            self.space_hash = 0

        self.set_extensions()

    @staticmethod
    def set_extensions():
        if not Span.has_extension("normalized_variant"):
            Span.set_extension("normalized_variant", getter=get_normalized_variant)

    def build_patterns(self, nlp: Language, terms: Patterns, progress: bool = False):
        """
        Build patterns and adds them for matching.
        Helper function for pipelines using this matcher.

        Parameters
        ----------
        nlp : Language
            The instance of the spaCy language class.
        terms : Patterns
            Dictionary of label/terms, or label/dictionary of terms/attribute.
        progress: bool
            Whether to track progress when preprocessing terms
        """

        if not terms:
            terms = dict()

        token_pipelines = [
            name
            for name, pipe in nlp.pipeline
            if any(
                "token" in assign and not assign == "token.is_sent_start"
                for assign in nlp.get_pipe_meta(name).assigns
            )
        ]
        with nlp.select_pipes(enable=token_pipelines):
            for key, expressions in (tqdm(
                terms.items(),
                desc="Adding terms into the pipeline"
            ) if progress else terms.items()):
                if isinstance(expressions, dict):
                    attr = expressions.get("attr")
                    expressions = expressions.get("patterns")
                else:
                    attr = None
                if isinstance(expressions, str):
                    expressions = [expressions]
                patterns = list(nlp.pipe(expressions))
                self.add(key, patterns, attr)

    cdef void find_matches(self, Doc doc, int start_idx, int end_idx, vector[SpanC] *matches) nogil:
        cdef:
            MapStruct * current_node = self.c_map
            int start = 0
            int idx = start_idx
            int idy = start_idx
            key_t key
            void * value
            int i = 0
            SpanC ms
            void * result
        while idx < end_idx:
            while (
                (0 < self.space_hash == doc.c[idx].tag)
                or (0 < self.excluded_hash == doc.c[idx].tag)
            ):
                idx += 1
                if idx >= end_idx:
                    return
            start = idx
            # look for sequences from this position
            token = Token.get_struct_attr(&doc.c[idx], self.attr)
            result = map_get(current_node, token)
            if result:
                current_node = <MapStruct *> result
                idy = idx + 1
                while idy < end_idx:
                    result = map_get(current_node, self._terminal_hash)
                    if result:
                        i = 0
                        while map_iter(<MapStruct *> result, &i, &key, &value):
                            ms = make_spanstruct(key, start, idy)
                            matches.push_back(ms)

                    while idy < end_idx and (
                        (0 < self.space_hash == doc.c[idy].tag)
                        or (0 < self.excluded_hash == doc.c[idy].tag)
                    ):
                        idy += 1
                    inner_token = Token.get_struct_attr(&doc.c[idy], self.attr)
                    result = map_get(current_node, inner_token)
                    if result:
                        current_node = <MapStruct *> result
                        idy += 1
                    else:
                        break
                else:
                    # end of doc reached
                    result = map_get(current_node, self._terminal_hash)
                    if result:
                        i = 0
                        while map_iter(<MapStruct *> result, &i, &key, &value):
                            ms = make_spanstruct(key, start, idy)
                            matches.push_back(ms)
            current_node = self.c_map
            idx += 1

cdef SpanC make_spanstruct(attr_t label, int start, int end) nogil:
    cdef SpanC spanc
    spanc.label = label
    spanc.start = start
    spanc.end = end
    return spanc
