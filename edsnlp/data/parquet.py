import os
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, TypeVar, Union

import pyarrow.dataset
import pyarrow.fs
import pyarrow.parquet
from pyarrow.dataset import ParquetFileFragment

from edsnlp import registry
from edsnlp.core.lazy_collection import LazyCollection
from edsnlp.data.base import BaseReader, BaseWriter
from edsnlp.data.converters import (
    FILENAME,
    get_dict2doc_converter,
    get_doc2dict_converter,
)
from edsnlp.utils.collections import dl_to_ld, flatten_once, ld_to_dl


class ParquetReader(BaseReader):
    DATA_FIELDS = ("dataset",)

    def __init__(
        self,
        path: Union[str, Path],
        *,
        read_in_worker: bool,
    ):
        super().__init__()
        path = (
            path
            if isinstance(path, Path) or "://" in path
            else f"file://{os.path.abspath(path)}"
        )
        self.read_in_worker = read_in_worker
        self.dataset = pyarrow.dataset.dataset(path, format="parquet")

    def read_main(self):
        fragments: List[ParquetFileFragment] = self.dataset.get_fragments()
        if self.read_in_worker:
            # read in worker -> each task is a file to read from
            return ((f, f.metadata.num_rows) for f in fragments)
        else:
            # read in worker -> each task is a non yet parsed line
            return (
                (line, 1)
                for f in fragments
                for batch in f.to_table().to_batches(1024)
                for line in dl_to_ld(batch.to_pydict())
            )

    def read_worker(self, tasks):
        if self.read_in_worker:
            tasks = list(
                chain.from_iterable(
                    dl_to_ld(batch.to_pydict())
                    for task in tasks
                    for batch in task.to_table().to_batches(1024)
                )
            )
        return tasks


T = TypeVar("T")


class ParquetWriter(BaseWriter):
    def __init__(
        self,
        path: Union[str, Path],
        num_rows_per_file: int,
        overwrite: bool,
        write_in_worker: bool,
        accumulate: bool = True,
    ):
        super().__init__()
        path = (
            path
            if isinstance(path, Path) or "://" in path
            else f"file://{os.path.abspath(path)}"
        )
        fs, fs_path = pyarrow.fs.FileSystem.from_uri(path)
        fs: pyarrow.fs.FileSystem
        fs.create_dir(fs_path, recursive=True)
        if overwrite is False:
            dataset = pyarrow.dataset.dataset(fs_path, format="parquet", filesystem=fs)
            if len(list(dataset.get_fragments())):
                raise FileExistsError(
                    f"Directory {path} already exists and is not empty. "
                    "Use overwrite=True to overwrite."
                )
        self.path = path
        self.write_in_worker = write_in_worker
        self.batch = []
        self.num_rows_per_file = num_rows_per_file
        self.closed = False
        self.finalized = False
        self.accumulate = accumulate
        if not self.accumulate:
            self.finalize = super().finalize

    def write_worker(self, records, last=False):
        # Results will contain a batches of samples ready to be written (or None if
        # write_in_worker is True) and they have already been written.
        results = []
        count = 0

        for rec in records:
            if isinstance(rec, dict):
                rec.pop(FILENAME, None)

        # While there is something to write
        greedy = last or not self.accumulate
        while len(records) or greedy and len(self.batch):
            n_to_fill = self.num_rows_per_file - len(self.batch)
            self.batch.extend(records[:n_to_fill])
            records = records[n_to_fill:]
            if greedy or len(self.batch) >= self.num_rows_per_file:
                fragment = pyarrow.Table.from_pydict(ld_to_dl(self.batch))  # type: ignore
                count += len(self.batch)
                self.batch = []
                if self.write_in_worker:
                    pyarrow.parquet.write_to_dataset(
                        table=fragment,
                        root_path=self.path,
                    )
                    fragment = None
                results.append(fragment)
        return results, count

    def finalize(self):
        if not self.finalized:
            self.finalized = True
            return self.write_worker([], last=True)

    def write_main(self, fragments: Iterable[List[Union[pyarrow.Table, Path]]]):
        fragments = list(fragments)
        for table in flatten_once(fragments):
            if not self.write_in_worker:
                pyarrow.parquet.write_to_dataset(
                    table=table,
                    root_path=self.path,
                )
        return pyarrow.dataset.dataset(self.path)


@registry.readers.register("parquet")
def read_parquet(
    path: Union[str, Path],
    converter: Union[str, Callable],
    *,
    read_in_worker: bool = False,
    **kwargs,
) -> LazyCollection:
    """
    The ParquetReader (or `edsnlp.data.read_parquet`) reads a directory of parquet files
    (or a single file) and yields documents.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)
    doc_iterator = edsnlp.data.read_parquet("path/to/parquet", converter="omop")
    annotated_docs = nlp.pipe(doc_iterator)
    ```

    !!! note "Generator vs list"

        `edsnlp.data.read_parquet` returns a
        [LazyCollection][edsnlp.core.lazy_collection.LazyCollection].
        To iterate over the documents multiple times efficiently or to access them by
        index, you must convert it to a list

        ```{ .python .no-check }
        docs = list(edsnlp.data.read_parquet("path/to/parquet", converter="omop"))
        ```

    Parameters
    ----------
    path: Union[str, Path]
        Path to the directory containing the parquet files (will recursively look for
        files in subdirectories). Supports any filesystem supported by pyarrow.
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the parquet rows of the data source to Doc objects
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
        reader=ParquetReader(
            path,
            read_in_worker=read_in_worker,
        )
    )
    if converter:
        converter, kwargs = get_dict2doc_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)
    return data


@registry.writers.register("parquet")
def write_parquet(
    data: Union[Any, LazyCollection],
    path: Union[str, Path],
    *,
    write_in_worker: bool = False,
    num_rows_per_file: int = 1024,
    overwrite: bool = False,
    accumulate: bool = True,
    converter: Optional[Union[str, Callable]],
    **kwargs,
) -> None:
    """
    `edsnlp.data.write_parquet` writes a list of documents as a parquet dataset.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)

    doc = nlp("My document with entities")

    edsnlp.data.write_parquet([doc], "path/to/parquet")
    ```

    !!! warning "Overwriting files"

        By default, `write_parquet` will raise an error if the directory already exists
        and contains parquet files. This is to avoid overwriting existing annotations.
        To allow overwriting existing files, use `overwrite=True`.

    Parameters
    ----------
    data: Union[Any, LazyCollection],
        The data to write (either a list of documents or a LazyCollection).
    path: Union[str, Path]
        Path to the directory containing the parquet files (will recursively look for
        files in subdirectories). Supports any filesystem supported by pyarrow.
    num_rows_per_file: int
        The maximum number of documents to write in each parquet file.
    overwrite: bool
        Whether to overwrite existing directories.
    write_in_worker: bool
        Whether to write the files in the workers or in the main process.
    accumulate: bool
        Whether to accumulate the results sent to the writer by workers until the
        batch is full or the writer is finalized. If False, each file will not be larger
        than the size of the batches it receives. This option requires that the writer
        is finalized before the end of the processing, which may not be compatible with
        some backends, such as `spark`.

        If `write_in_worker` is True, documents will be accumulated in each worker but
        not across workers, therefore leading to a larger number of files.
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

    return data.write(
        ParquetWriter(
            path,
            num_rows_per_file=num_rows_per_file,
            overwrite=overwrite,
            write_in_worker=write_in_worker,
            accumulate=accumulate,
        )
    )
