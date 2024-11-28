import json
import os
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union

from loguru import logger
from typing_extensions import Literal

from edsnlp import registry
from edsnlp.core.stream import Stream
from edsnlp.data.base import BaseWriter, FileBasedReader
from edsnlp.data.converters import (
    FILENAME,
    get_dict2doc_converter,
    get_doc2dict_converter,
)
from edsnlp.utils.collections import flatten, shuffle
from edsnlp.utils.file_system import FileSystem, normalize_fs_path, walk_match
from edsnlp.utils.stream_sentinels import DatasetEndSentinel
from edsnlp.utils.typing import AsList


class JsonReader(FileBasedReader):
    DATA_FIELDS = ()

    def __init__(
        self,
        path: Union[str, Path],
        *,
        keep_ipynb_checkpoints: bool,
        filesystem: Optional[FileSystem] = None,
        shuffle: Literal["dataset", False] = False,
        seed: Optional[int] = None,
        loop: bool = False,
        write_in_worker: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.write_in_worker = write_in_worker
        self.emitted_sentinels = {"dataset"}
        self.loop = loop
        self.fs, self.path = normalize_fs_path(filesystem, path)
        self.files = sorted(
            [
                file
                for file in walk_match(self.fs, self.path, ".*[.]json.*")
                if keep_ipynb_checkpoints or ".ipynb_checkpoints" not in str(file)
            ]
            if self.fs.isdir(self.path)
            else [self.path]
        )
        self.keep_ipynb_checkpoints = keep_ipynb_checkpoints
        self.shuffle = shuffle
        self.loop = loop
        seed = seed if seed is not None else random.getrandbits(32)
        self.rng = random.Random(seed)
        for file in self.files:
            if not self.fs.exists(file):
                raise FileNotFoundError(f"File {file} does not exist")
        assert len(self.files), f"No .json* file found under {self.path}"
        logger.info(
            f"Found {len(self.files)} file{'s' if len(self.files) > 1 else ''} "
            f"under {self.path}"
        )

    # TODO: implement read in worker = True / False

    def read_file(self, file: str):
        try:
            with self.fs.open(file, "r", encoding="utf8") as f:
                is_jsonl = os.path.splitext(file)[1].startswith(".jsonl")
                records = (
                    [{**json.loads(line), FILENAME: file} for line in f]
                    if is_jsonl
                    else [json.loads(f.read())]
                )
                return records
        except Exception as e:
            raise Exception(f"Cannot read {file}: {e}")

    def read_records(self) -> Iterable[Any]:
        while True:
            files = list(self.files)
            records = (line for file in files for line in self.read_file(file))
            if self.shuffle == "dataset":
                records = shuffle(list(records), self.rng)
            yield from records
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


T = TypeVar("T")


class JsonWriter(BaseWriter):
    def __init__(
        self,
        path: Union[str, Path],
        *,
        lines: Optional[bool] = None,
        overwrite: bool = False,
        filesystem: Optional[FileSystem] = None,
    ):
        self.fs, self.path = normalize_fs_path(filesystem, path)

        path_exists = self.fs.exists(self.path)
        path_is_file = path_exists and self.fs.isfile(self.path)
        might_be_file = Path(self.path).suffix != "" or path_is_file

        lines = lines if lines is not None else might_be_file
        self.lines = lines

        assert not path_exists or (might_be_file == lines), (
            f"To save as a single jsonl file, the path must be a file and lines "
            f"must be True. To save as a directory of json files, the path must "
            f"be a directory and lines must be False. "
            f"Got path={self.path} and lines={lines}."
        )
        if path_exists:
            if self.fs.isdir(self.path):
                files = [f for f in walk_match(self.fs, self.path, ".*[.]json.*")]
                if files:
                    if not overwrite:
                        raise FileExistsError(
                            f"Directory {self.path} already exists and appear to "
                            f"contain annotations:"
                            + "".join(
                                f"\n - {s[1:]}: {v} files"
                                for s, v in Counter(
                                    Path(f).suffix for f in files
                                ).items()
                            )
                            + "\nUse overwrite=True to write anyway."
                        )
                    for f in files:
                        self.fs.rm_file(f)

            elif not overwrite:
                raise FileExistsError(
                    f"File {self.path} already exists. Use overwrite=True to write "
                    "anyway."
                )

        if might_be_file != lines:
            warnings.warn(
                f"You set lines to {lines} but the path ({self.path}) you provided "
                f"looks like a {'file' if might_be_file else 'directory'}. "
                f"To save your documents as a single jsonl file, the path must be a "
                f"file and lines must be True. To save as a directory of json files, "
                f"the path must be a directory and lines must be False. "
            )

        super().__init__()

    def handle_record(self, record: Union[Dict, List[Dict]]):
        if self.lines:
            # If write as jsonl, we will perform the actual writing in the
            # `consolidate` method
            return record
        else:
            results = []
            for rec in flatten(record):
                filename = rec.pop(FILENAME, None)
                if filename is None:
                    raise KeyError(
                        "Cannot write to a directory of json files if the "
                        "FILENAME field is not present in the records. This is likely "
                        "caused by the `note_id` attribute (used as the filename stem) "
                        "not being set on the doc."
                    )

                file_path = os.path.join(self.path, f"{filename}.json")
                self.fs.makedirs(os.path.dirname(file_path), exist_ok=True)
                with self.fs.open(file_path, "w", encoding="utf-8") as f:
                    json.dump(rec, f)
                results.append(file_path)
            return results

    def consolidate(self, items: Union[Iterable[Path], Iterable[Dict]]):
        if self.lines:
            with self.fs.open(self.path, "w", encoding="utf-8") as f:
                out = []
                for record in flatten(items):
                    record.pop(FILENAME, None)
                    out.append(json.dumps(record))
                f.write("\n".join(out))
            return [self.path]
        else:
            items: Iterable[Path]
            return list(items)


@registry.readers.register("json")
def read_json(
    path: Union[str, Path],
    converter: Optional[AsList[Union[str, Callable]]] = None,
    *,
    keep_ipynb_checkpoints: bool = False,
    filesystem: Optional[FileSystem] = None,
    shuffle: Literal["dataset", False] = False,
    loop: bool = False,
    seed: int = 42,
    **kwargs,
) -> Stream:
    """
    The JsonReader (or `edsnlp.data.read_json`) reads a directory of JSON files and
    yields documents. At the moment, only entities and attributes are loaded.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)
    doc_iterator = edsnlp.data.read_json("path/to/json/dir", converter="omop")
    annotated_docs = nlp.pipe(doc_iterator)
    ```

    !!! note "Generator vs list"

        `edsnlp.data.read_json` returns a
        [Stream][edsnlp.core.stream.Stream].
        To iterate over the documents multiple times efficiently or to access them by
        index, you must convert it to a list

        ```{ .python .no-check }
        docs = list(edsnlp.data.read_json("path/to/json/dir", converter="omop")
        ```

    Parameters
    ----------
    path: Union[str, Path]
        Path to the directory containing the JSON files (will recursively look for
        files in subdirectories).
    keep_ipynb_checkpoints: bool
        Whether to keep the files have ".ipynb_checkpoints" in their path.
    filesystem: Optional[FileSystem]
        The filesystem to use to write the files. If None, the filesystem will be
        inferred from the path (e.g. `s3://` will use S3).
    shuffle: Literal["dataset", False]
        Whether to shuffle the data. If "dataset", the whole dataset will be shuffled
        before starting iterating on it (at the start of every epoch if looping).
    seed: Optional[int]
        The seed to use for shuffling.
    loop: bool
        Whether to loop over the data indefinitely.
    converter: Optional[AsList[Union[str, Callable]]]
        Converters to use to convert the JSON objects to Doc objects.
        These are documented on the [Converters](/data/converters) page.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.

    Returns
    -------
    Stream
    """
    if "read_in_worker" in kwargs:
        warnings.warn(
            "The `read_in_worker` parameter of "
            "edsnlp.data.read_parquet is deprecated and set "
            "to True by default.",
            FutureWarning,
        )

    data = Stream(
        reader=JsonReader(
            path,
            keep_ipynb_checkpoints=keep_ipynb_checkpoints,
            filesystem=filesystem,
            shuffle=shuffle,
            seed=seed,
            loop=loop,
            **{k: kwargs.pop(k) for k in ("read_in_worker",) if k in kwargs},
        )
    )
    if converter:
        for conv in converter:
            conv, kwargs = get_dict2doc_converter(conv, kwargs)
            data = data.map(conv, kwargs=kwargs)
    return data


@registry.writers.register("json")
def write_json(
    data: Union[Any, Stream],
    path: Union[str, Path],
    *,
    lines: bool = None,
    overwrite: bool = False,
    execute: bool = True,
    converter: Optional[Union[str, Callable]] = None,
    filesystem: Optional[FileSystem] = None,
    **kwargs,
) -> None:
    """
    `edsnlp.data.write_json` writes a list of documents using the JSON
    format in a directory. If `lines` is false, each document will be stored in its
    own JSON file, named after the FILENAME field returned by the converter (commonly
    the `note_id` attribute of the documents), and subdirectories will be created if the
    name contains `/` characters.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)

    doc = nlp("My document with entities")

    edsnlp.data.write_json([doc], "path/to/json/file", converter="omop", lines=True)
    # or to write a directory of JSON files, ensure that each doc has a doc._.note_id
    # attribute, since this will be used as a filename:
    edsnlp.data.write_json([doc], "path/to/json/dir", converter="omop", lines=False)
    ```

    !!! warning "Overwriting files"

        By default, `write_json` will raise an error if the directory already exists
        and contains files with `.a*` or `.txt` suffixes. This is to avoid overwriting
        existing annotations. To allow overwriting existing files, use `overwrite=True`.

    Parameters
    ----------
    data: Union[Any, Stream],
        The data to write (either a list of documents or a Stream).
    path: Union[str, Path]
        Path to either
        - a file if `lines` is true : this will write the documents as a JSONL file
        - a directory if `lines` is false: this will write one JSON file per document
          using the FILENAME field returned by the converter (commonly the `note_id`
          attribute of the documents) as the filename.
    lines: Optional[bool]
        Whether to write the documents as a JSONL file or as a directory of JSON files.
        By default, this is inferred from the path: if the path is a file, lines is
        assumed to be true, otherwise it is assumed to be false.
    overwrite: bool
        Whether to overwrite existing directories.
    execute: bool
        Whether to execute the writing operation immediately or to return a stream
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the documents to dictionary objects before writing
        them. These are documented on the [Converters](/data/converters) page.
    filesystem: Optional[FileSystem]
        The filesystem to use to write the files. If None, the filesystem will be
        inferred from the path (e.g. `s3://` will use S3).
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.
    """

    data = Stream.ensure_stream(data)
    if converter:
        converter, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)

    return data.write(
        JsonWriter(
            path,
            lines=lines,
            overwrite=overwrite,
            filesystem=filesystem,
        ),
        execute=execute,
    )
