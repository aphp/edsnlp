import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, TypeVar, Union

from loguru import logger

from edsnlp import registry
from edsnlp.core.lazy_collection import LazyCollection
from edsnlp.data.base import BaseReader, BaseWriter
from edsnlp.data.converters import (
    FILENAME,
    get_dict2doc_converter,
    get_doc2dict_converter,
)
from edsnlp.utils.collections import flatten_once


class JsonReader(BaseReader):
    DATA_FIELDS = ("path",)

    def __init__(
        self,
        path: Union[str, Path],
        *,
        keep_ipynb_checkpoints: bool,
        read_in_worker: bool,
    ):
        super().__init__()
        self.path = Path(path)
        self.keep_ipynb_checkpoints = keep_ipynb_checkpoints
        self.read_in_worker = read_in_worker
        self.files = (
            [
                file
                for file in self.path.rglob("*.json*")
                if self.keep_ipynb_checkpoints or ".ipynb_checkpoints" not in str(file)
            ]
            if self.path.is_dir()
            else [self.path]
        )
        for file in self.files:
            if not file.exists():
                raise FileNotFoundError(f"File {file} does not exist")
        assert len(self.files), f"No .json* file found under {path}"
        logger.info(
            f"Found {len(self.files)} file{'s' if len(self.files) > 1 else ''} "
            f"under {path}"
        )

    # TODO: implement read in worker = True / False

    def read_file(self, file: Path):
        return (
            Path(file).read_text().splitlines()
            if file.suffix.startswith(".jsonl")
            else [Path(file).read_text()]
        )

    def read_main(self):
        if self.read_in_worker:
            # read in worker -> each task is a file to read from
            return ((f, 0 if f.suffix.startswith(".jsonl") else 1) for f in self.files)
        else:
            # read in worker -> each task is a non yet parsed line
            return (
                ((f, line, f.suffix.startswith(".jsonl")), 1)
                for f in self.files
                for line in self.read_file(f)
            )

    def read_worker(self, tasks):
        results = []
        for task in tasks:
            if self.read_in_worker:
                filename = task
                content = self.read_file(filename)
                is_jsonl = filename.suffix.startswith(".jsonl")
            else:
                filename, content, is_jsonl = task
                content = [content]
            try:
                for line in content:
                    obj = json.loads(line)
                    if not is_jsonl:
                        obj[FILENAME] = filename
                    results.append(obj)
            except Exception:
                raise Exception(f"Cannot parse {filename}")
        return results


T = TypeVar("T")


class JsonWriter(BaseWriter):
    def __init__(
        self,
        path: Union[str, Path],
        *,
        lines: bool = True,
        overwrite: bool = False,
    ):
        self.path = Path(path)
        self.lines = lines

        if self.path.exists() and self.path.is_dir():
            exts = Counter(f.suffix for f in self.path.iterdir())
            unsafe_exts = {s[1:]: v for s, v in exts.items() if s.startswith(".json")}
            if unsafe_exts and not overwrite:
                raise FileExistsError(
                    f"Directory {self.path} already exists and appear to contain "
                    "annotations:"
                    + "".join(f"\n - {s}: {v} files" for s, v in unsafe_exts.items())
                    + "\nUse overwrite=True to write files anyway."
                )

        super().__init__()

    def write_worker(self, records):
        # If write as jsonl, we will perform the actual writing in the `write` method
        if self.lines:
            results = []
            for rec in records:
                rec.pop(FILENAME, None)
                results.append(json.dumps(rec))
            return results, len(results)
        else:
            results = []
            for rec in records:
                file_path = self.path / f"{str(rec.pop(FILENAME))}.json"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w") as f:
                    json.dump(rec, f)
                results.append(file_path)
            return results, len(results)

    def write_main(self, fragments: Iterable[Union[List[Path], List[str]]]):
        fragments = list(flatten_once(fragments))
        if self.lines:
            self.path.write_text("\n".join(fragments))
            return [self.path]
        else:
            return [f for f in fragments]


@registry.readers.register("json")
def read_json(
    path: Union[str, Path],
    converter: Union[str, Callable],
    *,
    keep_ipynb_checkpoints: bool = False,
    read_in_worker: bool = False,
    **kwargs,
) -> LazyCollection:
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
        [LazyCollection][edsnlp.core.lazy_collection.LazyCollection].
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
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the JSON rows of the data source to Doc objects
    keep_ipynb_checkpoints: bool
        Whether to keep the files have ".ipynb_checkpoints" in their path.
    read_in_worker: bool
        Whether to read the files in the worker or in the main process.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented
        on the [Data schemas](/data/schemas) page.

    Returns
    -------
    LazyCollection
    """
    data = LazyCollection(
        reader=JsonReader(
            path,
            keep_ipynb_checkpoints=keep_ipynb_checkpoints,
            read_in_worker=read_in_worker,
        )
    )
    if converter:
        converter, kwargs = get_dict2doc_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)
    return data


@registry.writers.register("json")
def write_json(
    data: Union[Any, LazyCollection],
    path: Union[str, Path],
    *,
    lines: bool = True,
    overwrite: bool = False,
    converter: Optional[Union[str, Callable]],
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

    edsnlp.data.write_json([doc], "path/to/json/dir", converter="omop")
    ```

    !!! warning "Overwriting files"

        By default, `write_json` will raise an error if the directory already exists
        and contains files with `.a*` or `.txt` suffixes. This is to avoid overwriting
        existing annotations. To allow overwriting existing files, use `overwrite=True`.

    Parameters
    ----------
    data: Union[Any, LazyCollection],
        The data to write (either a list of documents or a LazyCollection).
    path: Union[str, Path]
        Path to the directory containing the JSON files (will recursively look for
        files in subdirectories).
    lines: bool
        Whether to write the documents as a JSONL file (default).
    overwrite: bool
        Whether to overwrite existing directories.
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the documents to dictionary objects before writing
        them.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented
        on the [Data schemas](/data/schemas) page.
    """

    data = LazyCollection.ensure_lazy(data)
    if converter:
        converter, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)

    return data.write(JsonWriter(path, lines=lines, overwrite=overwrite))
