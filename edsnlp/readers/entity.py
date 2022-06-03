from functools import partial
from pathlib import Path
from typing import Callable, Iterable

import spacy
from spacy.language import Language
from spacy.tokens import Doc, DocBin
from spacy.training import Example


@spacy.registry.readers("edsnlp.ents_corpus.v1")
def create_docbin_reader(file: Path) -> Callable[["Language"], Iterable[Example]]:
    return partial(read_files, file)


def read_files(file: Path, nlp: "Language") -> Iterable[Example]:
    """Custom reader that keeps the tokenization of the gold data,
    and also adds the gold GGP annotations as we do not attempt to predict these."""
    doc_bin = DocBin().from_disk(file)
    docs = doc_bin.get_docs(nlp.vocab)
    for gold in docs:
        pred = Doc(
            nlp.vocab,
            words=[t.text for t in gold],
            spaces=[bool(t.whitespace_) for t in gold],
        )
        pred.ents = gold.ents
        yield Example(pred, gold)
