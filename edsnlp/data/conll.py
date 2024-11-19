import os
import random
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from fsspec import filesystem as fsspec
from loguru import logger
from typing_extensions import Literal

from edsnlp import registry
from edsnlp.core.stream import Stream
from edsnlp.data.base import FileBasedReader
from edsnlp.data.converters import FILENAME, get_dict2doc_converter
from edsnlp.utils.collections import shuffle
from edsnlp.utils.file_system import FileSystem, normalize_fs_path, walk_match
from edsnlp.utils.stream_sentinels import DatasetEndSentinel
from edsnlp.utils.typing import AsList

LOCAL_FS = fsspec("file")

DEFAULT_COLUMNS = [
    "ID",
    "FORM",
    "LEMMA",
    "UPOS",
    "XPOS",
    "FEATS",
    "HEAD",
    "DEPREL",
    "DEPS",
    "MISC",
]


def parse_conll(
    path: str,
    cols: Optional[List[str]] = None,
    fs: FileSystem = LOCAL_FS,
) -> Iterable[Dict]:
    """
    Load a .conll file and return a dictionary with the text, words, and entities.
    This expects the file to contain multiple sentences, split into words, each one
    described in a line. Each sentence is separated by an empty line.

    If possible, looks for a `#global.columns` comment at the start of the file to
    extract the column names.

    Examples:

    ```text
    ...
    11	jeune	jeune	ADJ	_	Number=Sing	12	amod	_	_
    12	fille	fille	NOUN	_	Gender=Fem|Number=Sing	5	obj	_	_
    13	qui	qui	PRON	_	PronType=Rel	14	nsubj	_	_
    ...
    ```

    Parameters
    ----------
    path: str
        Path or glob path of the brat text file (.txt, not .ann)
    cols: Optional[List[str]]
        List of column names to use. If None, the first line of the file will be used
    fs: FileSystem
        Filesystem to use

    Returns
    -------
    Iterator[Dict]
    """
    with fs.open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if cols is None:
        try:
            cols = next(
                line.split("=")[1].strip().split()
                for line in lines
                if line.strip("# ").startswith("global.columns")
            )
        except StopIteration:
            cols = DEFAULT_COLUMNS
            warnings.warn(
                f"No #global.columns comment found in the CoNLL file. "
                f"Using default {cols}"
            )

    doc = {"words": []}
    for line in lines:
        line = line.strip()
        if not line:
            if doc["words"]:
                yield doc
                doc = {"words": []}
            continue
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        word = {k: v for k, v in zip(cols, parts) if v != "_"}
        doc["words"].append(word)

    if doc["words"]:
        yield doc


class ConllReader(FileBasedReader):
    DATA_FIELDS = ()

    def __init__(
        self,
        path: Union[str, Path],
        *,
        columns: Optional[List[str]] = None,
        filesystem: Optional[FileSystem] = None,
        loop: bool = False,
        shuffle: Literal["dataset", False] = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.emitted_sentinels = {"dataset"}
        self.rng = random.Random(seed)
        self.loop = loop
        self.fs, self.path = normalize_fs_path(filesystem, path)
        self.columns = columns

        files = walk_match(self.fs, self.path, ".*[.]conllu?")
        self.files = sorted(files)
        assert len(self.files), f"No .conll files found in the directory {self.path}"
        logger.info(f"The directory contains {len(self.files)} .conll files.")

    def read_records(self) -> Iterable[Any]:
        while True:
            files = self.files
            if self.shuffle:
                files = shuffle(files, self.rng)
            for item in files:
                for anns in parse_conll(item, cols=self.columns, fs=self.fs):
                    anns[FILENAME] = os.path.relpath(item, self.path).rsplit(".", 1)[0]
                    anns["doc_id"] = anns[FILENAME]
                    yield anns
            yield DatasetEndSentinel()
            if not self.loop:
                break

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"path={self.path!r}, "
            f"shuffle={self.shuffle}, "
            f"loop={self.loop})"
        )


# No writer for CoNLL format yet


@registry.readers.register("conll")
def read_conll(
    path: Union[str, Path],
    *,
    columns: Optional[List[str]] = None,
    converter: Optional[AsList[Union[str, Callable]]] = ["conll"],
    filesystem: Optional[FileSystem] = None,
    shuffle: Literal["dataset", False] = False,
    seed: Optional[int] = None,
    loop: bool = False,
    **kwargs,
) -> Stream:
    """
    The ConllReader (or `edsnlp.data.read_conll`) reads a file or directory of CoNLL
    files and yields documents.

    The raw output (i.e., by setting `converter=None`) will be in the following form
    for a single doc:

    ```
    {
        "words": [
            {"ID": "1", "FORM": ...},
            ...
        ],
    }
    ```

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)
    doc_iterator = edsnlp.data.read_conll("path/to/conll/file/or/directory")
    annotated_docs = nlp.pipe(doc_iterator)
    ```

    !!! note "Generator vs list"

        `edsnlp.data.read_conll` returns a
        [Stream][edsnlp.core.stream.Stream].
        To iterate over the documents multiple times efficiently or to access them by
        index, you must convert it to a list :

        ```{ .python .no-check }
        docs = list(edsnlp.data.read_conll("path/to/conll/file/or/directory"))
        ```

    Parameters
    ----------
    path : Union[str, Path]
        Path to the directory containing the CoNLL files (will recursively look for
        files in subdirectories).
    columns: Optional[List[str]]
        List of column names to use. If None, will try to extract to look for a
        `#global.columns` comment at the start of the file to extract the column names.
    shuffle: Literal["dataset", False]
        Whether to shuffle the data. If "dataset", the whole dataset will be shuffled
        before starting iterating on it (at the start of every epoch if looping).
    seed: Optional[int]
        The seed to use for shuffling.
    loop: bool
        Whether to loop over the data indefinitely.
    nlp : Optional[PipelineProtocol]
        The pipeline object (optional and likely not needed, prefer to use the
        `tokenizer` directly argument instead).
    tokenizer : Optional[spacy.tokenizer.Tokenizer]
        The tokenizer instance used to tokenize the documents. Likely not needed since
        by default it uses the current context tokenizer :

        - the tokenizer of the next pipeline run by `.map_pipeline` in a
          [Stream][edsnlp.core.stream.Stream].
        - or the `eds` tokenizer by default.
    converter : Optional[AsList[Union[str, Callable]]]
        Converter to use to convert the documents to dictionary objects.
    filesystem: Optional[FileSystem] = None,
        The filesystem to use to write the files. If None, the filesystem will be
        inferred from the path (e.g. `s3://` will use S3).
    """

    data = Stream(
        reader=ConllReader(
            path,
            columns=columns,
            filesystem=filesystem,
            loop=loop,
            shuffle=shuffle,
            seed=seed,
        )
    )
    if converter:
        for conv in converter:
            conv, kwargs = get_dict2doc_converter(conv, kwargs)
            data = data.map(conv, kwargs=kwargs)
    return data
