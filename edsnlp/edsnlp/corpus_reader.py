import random
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Union

import spacy
from spacy import Errors, Vocab
from spacy.language import Language
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Corpus, Example, dont_augment
from spacy.training.corpus import FILE_TYPE, walk_corpus

if not Doc.has_extension("context"):
    Doc.set_extension("context", default=dict())
if not Doc.has_extension("note_id"):
    Doc.set_extension("note_id", default=None)
if not Doc.has_extension("note_datetime"):
    Doc.set_extension("note_datetime", default=None)
if not Doc.has_extension("note_class_source_value"):
    Doc.set_extension("note_class_source_value", default=None)
if not Doc.has_extension("split"):
    Doc.set_extension("split", default=None)


@spacy.registry.readers("eds.Corpus.v1")
class Corpus(Corpus):
    def __init__(
        self,
        path: Union[str, Path],
        *,
        limit: int = 0,
        gold_preproc: bool = False,
        max_length: int = 0,
        augmenter: Optional[Callable] = None,
        shuffle: bool = False,
        filter_expr: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if path is None:
            raise ValueError(Errors.E913)
        spacy.util.logger.debug(f"Loading corpus from path: {path}")
        self.path = spacy.util.ensure_path(path)
        self.gold_preproc = gold_preproc
        self.max_length = max_length
        self.limit = limit
        self.augmenter = augmenter if augmenter is not None else dont_augment
        self.shuffle = shuffle
        self.filter_fn = eval(f"lambda doc: {filter_expr}") if filter_expr else None
        if filter_expr is not None:
            spacy.util.logger.info(f"Filtering corpus with expression: {filter_expr}")
        self.seed = seed

    def __call__(self, nlp: "Language") -> Iterator[Example]:
        """Yield examples from the data.

        A difference with the standard spacy.Corpus object is that we
        - first shuffle the data
        - then subset it

        nlp (Language): The current nlp object.
        YIELDS (Example): The examples.
        DOCS: https://spacy.io/api/corpus#call
        """
        ref_docs = self.read_docbin(nlp.vocab, walk_corpus(self.path, FILE_TYPE))
        if self.shuffle:
            ref_docs = list(ref_docs)  # type: ignore
            random.Random(self.seed).shuffle(ref_docs)  # type: ignore

            if self.limit >= 1:
                ref_docs = ref_docs[: self.limit]

        if self.gold_preproc:
            examples = self.make_examples_gold_preproc(nlp, ref_docs)
        else:
            examples = self.make_examples(nlp, ref_docs)
        for real_eg in examples:
            if len(real_eg) and len(real_eg.reference.ents):
                for augmented_eg in self.augmenter(
                    nlp, real_eg
                ):  # type: ignore[operator]
                    yield augmented_eg

    def subset_doc(self, doc, start, end):
        new_doc = doc[start:end].as_doc(copy_user_data=True)
        for name, group in doc.spans.items():
            new_doc.spans[name] = [
                Span(
                    new_doc,
                    max(0, span.start - start),
                    min(end, span.end) - start,
                    span.label,
                )
                for span in group
                if span.end > start and span.start < end
            ]

        return new_doc

    def make_examples(
        self, nlp: "Language", reference_docs: Iterable[Doc]
    ) -> Iterator[Example]:
        for reference in reference_docs:
            if len(reference) == 0:
                continue
            elif self.max_length == 0 or len(reference) < self.max_length:
                yield self._make_example(nlp, reference, False)
            else:
                start = 0
                end = 0
                for sent in (
                    reference.sents
                    if reference.has_annotation("SENT_START")
                    else (reference[:],)
                ):
                    if len(sent) == 0:
                        continue
                    # If the sentence adds too many tokens
                    if sent.end - start > self.max_length:
                        # But the current buffer too large
                        while end - start > self.max_length:
                            yield self._make_example(
                                nlp,
                                self.subset_doc(
                                    reference, start, start + self.max_length
                                ),
                                False,
                            )
                            start = start + self.max_length
                        yield self._make_example(
                            nlp, self.subset_doc(reference, start, end), False
                        )
                        start = end

                    # Otherwise, extend the current buffer
                    end = sent.end

                while end - start > self.max_length:
                    yield self._make_example(
                        nlp,
                        self.subset_doc(reference, start, start + self.max_length),
                        False,
                    )
                    start = start + self.max_length
                yield self._make_example(
                    nlp, self.subset_doc(reference, start, end), False
                )

    def _make_example(
        self, nlp: "Language", reference: Doc, gold_preproc: bool
    ) -> Example:
        eg = super()._make_example(nlp, reference, gold_preproc)
        eg.predicted._.note_id = eg.reference._.note_id
        eg.predicted._.note_datetime = eg.reference._.note_datetime
        eg.predicted._.note_class_source_value = eg.reference._.note_class_source_value
        eg.predicted._.context = eg.reference._.context
        eg.predicted._.split = eg.reference._.split

        eg.predicted.ents = eg.reference.ents
        for name in eg.predicted.spans:
            eg.predicted.spans[name] = eg.reference.spans[name]

        return eg

    def read_docbin(
        self, vocab: Vocab, locs: Iterable[Union[str, Path]]
    ) -> Iterator[Doc]:
        """Yield training examples as example dicts"""
        self.not_called_twice = False
        for loc in locs:
            loc = spacy.util.ensure_path(loc)
            if loc.parts[-1].endswith(FILE_TYPE):  # type: ignore[union-attr]
                doc_bin = DocBin().from_disk(loc)
                docs = doc_bin.get_docs(vocab)
                for doc in docs:
                    if len(doc) and (self.filter_fn is None or self.filter_fn(doc)):
                        yield doc
