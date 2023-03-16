import os
import pickle
import tempfile
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pysimstring.simstring as simstring
from spacy import Language, Vocab
from spacy.tokens import Doc, Span
from tqdm import tqdm

from edsnlp.matchers.utils import ATTRIBUTES, get_text


class SimstringWriter:
    def __init__(self, path: Union[str, Path]):
        """
        A context class to write a simstring database

        Parameters
        ----------
        path: Union[str, Path]
            Path to database
        """
        os.makedirs(path, exist_ok=True)
        self.path = path

    def __enter__(self):
        path = os.path.join(self.path, "terms.simstring")
        self.db = simstring.writer(path, 3, False, True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()

    def insert(self, term):
        self.db.insert(term)


class SimilarityMeasure(str, Enum):
    jaccard = "jaccard"
    dice = "dice"
    overlap = "overlap"
    cosine = "cosine"


class SimstringMatcher:
    def __init__(
        self,
        vocab: Vocab,
        path: Optional[Union[Path, str]] = None,
        measure: SimilarityMeasure = SimilarityMeasure.dice,
        threshold: float = 0.75,
        windows: int = 5,
        ignore_excluded: bool = False,
        ignore_space_tokens: bool = False,
        attr: str = "NORM",
    ):
        """
        PhraseMatcher that allows to skip excluded tokens.
        Heavily inspired by https://github.com/Georgetown-IR-Lab/QuickUMLS

        Parameters
        ----------
        vocab : Vocab
            spaCy vocabulary to match on.
        path: Optional[Union[Path, str]]
            Path where we will store the precomputed patterns
        measure: SimilarityMeasure
            Name of the similarity measure.
            One of [jaccard, dice, overlap, cosine]
        windows: int
            Maximum number of words in a candidate span
        threshold: float
            Minimum similarity value to match a concept's synonym
        ignore_excluded : Optional[bool]
            Whether to exclude tokens that have an EXCLUDED tag, by default False
        ignore_space_tokens : Optional[bool]
            Whether to exclude tokens that have a "SPACE" tag, by default False
        attr : str
            Default attribute to match on, by default "TEXT".
            Can be overridden in the `add` method.
            To match on a custom attribute, prepend the attribute name with `_`.
        """

        assert measure in (
            SimilarityMeasure.jaccard,
            SimilarityMeasure.dice,
            SimilarityMeasure.overlap,
            SimilarityMeasure.cosine,
        )

        self.vocab = vocab
        self.windows = windows
        self.measure = measure
        self.threshold = threshold
        self.ignore_excluded = ignore_excluded
        self.ignore_space_tokens = ignore_space_tokens
        self.attr = attr

        if path is None:
            path = tempfile.mkdtemp()
        self.path = Path(path)

        self.ss_reader = None
        self.syn2cuis = None

    def build_patterns(
        self, nlp: Language, terms: Dict[str, Iterable[str]], progress: bool = False
    ):
        """
        Build patterns and adds them for matching.

        Parameters
        ----------
        nlp : Language
            The instance of the spaCy language class.
        terms : Patterns
            Dictionary of label/terms, or label/dictionary of terms/attribute.
        progress: bool
            Whether to track progress when preprocessing terms
        """

        self.ss_reader = None
        self.syn2cuis = None

        syn2cuis = defaultdict(lambda: [])
        token_pipelines = [
            name
            for name, pipe in nlp.pipeline
            if any(
                "token" in assign and not assign == "token.is_sent_start"
                for assign in nlp.get_pipe_meta(name).assigns
            )
        ]
        with nlp.select_pipes(enable=token_pipelines):
            with SimstringWriter(self.path) as ss_db:
                for cui, synset in tqdm(terms.items()) if progress else terms.items():
                    for term in nlp.pipe(synset):
                        norm_text = get_text(
                            term,
                            self.attr,
                            ignore_excluded=self.ignore_excluded,
                            ignore_space_tokens=self.ignore_space_tokens,
                        )
                        term = "##" + norm_text + "##"
                        ss_db.insert(term)
                        syn2cuis[term].append(cui)
        syn2cuis = {term: tuple(sorted(set(cuis))) for term, cuis in syn2cuis.items()}
        with open(self.path / "cui-db.pkl", "wb") as f:
            pickle.dump(syn2cuis, f)

    def load(self):
        if self.ss_reader is None:
            self.ss_reader = simstring.reader(
                os.path.join(self.path, "terms.simstring")
            )
            self.ss_reader.measure = getattr(simstring, self.measure)
            self.ss_reader.threshold = self.threshold

            with open(os.path.join(self.path, "cui-db.pkl"), "rb") as f:
                self.syn2cuis = pickle.load(f)

    def __call__(self, doc, as_spans=False):
        self.load()

        root = getattr(doc, "doc", doc)
        if root.has_annotation("IS_SENT_START"):
            sents = tuple(doc.sents)
        else:
            sents = (doc,)

        ents: List[Tuple[str, int, int, float]] = []

        for sent in sents:
            text, offsets = get_text_and_offsets(
                doclike=sent,
                attr=self.attr,
                ignore_excluded=self.ignore_excluded,
            )
            sent_start = getattr(sent, "start", 0)
            for size in range(1, self.windows):
                for i in range(0, len(offsets) - size):
                    begin_char, _, begin_i, _ = offsets[i]
                    _, end_char, _, end_i = offsets[i + size]
                    span_text = "##" + text[begin_char:end_char] + "##"
                    matches = self.ss_reader.retrieve(span_text)
                    for res in matches:
                        sim = _similarity(span_text, res, measure=self.measure)
                        for cui in self.syn2cuis[res]:
                            ents.append(
                                (cui, begin_i + sent_start, end_i + sent_start, sim)
                            )

        sorted_spans = sorted(ents, key=simstring_sort_key, reverse=True)
        results = []
        seen_tokens = set()
        for span in sorted_spans:
            # Check for end - 1 here because boundaries are inclusive
            span_tokens = set(range(span[1], span[2]))
            if not (span_tokens & seen_tokens):
                results.append(span)
                seen_tokens.update(span_tokens)
        results = sorted(results, key=lambda span: span[1])
        if as_spans:
            spans = [
                Span(root, span_data[1], span_data[2], span_data[0])
                for span_data in results
            ]
            return spans
        else:
            return [(self.vocab.strings[span[0]], span[1], span[2]) for span in results]


def _similarity(x: str, y: str, measure: SimilarityMeasure = SimilarityMeasure.dice):

    x_ngrams = {x[i : i + 3] for i in range(0, len(x) - 3)}
    y_ngrams = {y[i : i + 3] for i in range(0, len(y) - 3)}

    if measure == SimilarityMeasure.jaccard:
        return len(x_ngrams & y_ngrams) / (len(x_ngrams | y_ngrams))

    if measure == SimilarityMeasure.dice:
        return 2 * len(x_ngrams & y_ngrams) / (len(x_ngrams) + len(y_ngrams))

    if measure == SimilarityMeasure.cosine:
        return len(x_ngrams & y_ngrams) / sqrt(len(x_ngrams) * len(y_ngrams))

    if measure == SimilarityMeasure.overlap:
        return len(x_ngrams & y_ngrams)


def simstring_sort_key(span_data: Tuple[str, int, int, float]):
    return span_data[3], span_data[2] - span_data[1], -span_data[1]


@lru_cache(maxsize=128)
def get_text_and_offsets(
    doclike: Union[Span, Doc],
    attr: str = "TEXT",
    ignore_excluded: bool = True,
    ignore_space_tokens: bool = True,
) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Align different representations of a `Doc` or `Span` object.

    Parameters
    ----------
    doclike : Doc
        spaCy `Doc` or `Span` object
    attr : str, optional
        Attribute to use, by default `"TEXT"`
    ignore_excluded : bool
        Whether to remove excluded tokens, by default True
    ignore_space_tokens : bool
        Whether to remove space tokens, by default False


    Returns
    -------
    Tuple[str, List[Tuple[int, int, int, int]]]
        The new clean text and offset tuples for each word giving the begin char indice
        of the word in the new text, the end char indice of its preceding word and the
        begin / end indices of the word in the original document
    """
    attr = attr.upper()
    attr = ATTRIBUTES.get(attr, attr)

    custom = attr.startswith("_")

    if custom:
        attr = attr[1:].lower()

    offsets = []

    cursor = 0

    text = []

    last = cursor
    last_i = 0
    for i, token in enumerate(doclike):

        if (not ignore_excluded or token.tag_ != "EXCLUDED") and (
            not ignore_space_tokens or token.tag_ != "SPACE"
        ):
            if custom:
                token_text = getattr(token._, attr)
            else:
                token_text = getattr(token, attr)

            # We add the cursor
            end = cursor + len(token_text)
            offsets.append((cursor, last, i, last_i + 1))

            cursor = end
            last = end
            last_i = i

            text.append(token_text)

            if token.whitespace_:
                cursor += 1
                text.append(" ")

    offsets.append((cursor, last, len(doclike), last_i + 1))

    return "".join(text), offsets
