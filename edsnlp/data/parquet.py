import random
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pyarrow.dataset
import pyarrow.fs
import pyarrow.parquet
from confit import VisibleDeprecationWarning
from pyarrow.dataset import ParquetFileFragment
from typing_extensions import Literal

from edsnlp import registry
from edsnlp.core.lazy_collection import LazyCollection
from edsnlp.data.base import BatchWriter, FileBasedReader
from edsnlp.data.converters import (
    get_dict2doc_converter,
    get_doc2dict_converter,
)
from edsnlp.utils.collections import batchify, dl_to_ld, flatten, ld_to_dl, shuffle
from edsnlp.utils.file_system import FileSystem, normalize_fs_path


class ParquetReader(FileBasedReader):
    DATA_FIELDS = ()

    def __init__(
        self,
        path: Union[str, Path],
        *,
        filesystem: Optional[FileSystem] = None,
        shuffle: Literal["dataset", "file", False] = False,
        seed: Optional[int] = None,
        loop: bool = False,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.loop = loop
        # Either the filesystem has not been passed
        # or the path is a URL (e.g. s3://) => we need to infer the filesystem
        self.fs, self.path = normalize_fs_path(filesystem, path)
        self.fragments: List[ParquetFileFragment] = list(
            pyarrow.dataset.dataset(
                self.path,
                format="parquet",
                filesystem=self.fs,
            ).get_fragments()
        )

    def read_fragment(self, fragment: ParquetFileFragment) -> Iterable[Dict]:
        return dl_to_ld(fragment.to_table().to_pydict())

    def read_records(self) -> Iterable[Any]:
        while True:
            files = list(self.fragments)
            if self.shuffle == "file":
                yield from (
                    line
                    for file in shuffle(files, self.rng)
                    for line in shuffle(list(self.read_fragment(file)), self.rng)
                )
            else:
                records = (line for file in files for line in self.read_fragment(file))
                if self.shuffle == "dataset":
                    yield from shuffle(
                        list(records),
                        self.rng,
                    )
                else:
                    yield from records
            if not self.loop:
                break

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"path={self.path!r}, "
            f"shuffle={self.shuffle}, "
            f"loop={self.loop})"
        )


class ParquetWriter(BatchWriter):
    def __init__(
        self,
        *,
        path: Union[str, Path],
        batch_size: Optional[Union[int]] = None,
        batch_by: Union[Callable, Literal["record"]] = "record",
        batch_in_worker: bool = False,
        overwrite: bool,
        filesystem: Optional[FileSystem] = None,
    ):
        super().__init__()
        self.fs, self.path = normalize_fs_path(filesystem, path)

        # Check that filesystem has the same protocol as indicated by path
        self.fs.makedirs(self.path, exist_ok=True)

        dataset: pyarrow.dataset.FileSystemDataset = (  # type: ignore
            pyarrow.dataset.dataset(
                self.path,
                format="parquet",
                filesystem=self.fs,
            )
        )
        if len(list(dataset.files)):
            if not overwrite:
                raise FileExistsError(
                    f"Directory {self.path} already exists and is not empty. "
                    "Use overwrite=True to overwrite."
                )
            for file in dataset.files:
                self.fs.rm_file(file)
        self.fs = filesystem
        self.batch_by = {
            "record": batchify,
            # "file": batchify_by_fragment,
        }.get(batch_by, batch_by)
        if (
            batch_by in ("record", "doc")
            or self.batch_by is batchify
            and batch_size is None
        ):
            warnings.warn(
                "You should specify a batch size when using record-wise batch writing. "
                "Setting batch size to 1024."
            )
            batch_size = 1024

        self.batch_size = batch_size
        self.batch_in_worker = batch_in_worker
        self.batch = []
        self.closed = False

    def handle_batch(self, batch: List[Dict]) -> Tuple[ParquetFileFragment, int]:
        fragment = pyarrow.Table.from_pydict(ld_to_dl(flatten(batch)))
        pyarrow.parquet.write_to_dataset(
            table=fragment,
            root_path=self.path,
            filesystem=self.fs,
        )
        return (fragment, len(batch))

    def consolidate(
        self, items: Iterable[ParquetFileFragment]
    ) -> pyarrow.dataset.Dataset:
        for _ in items:
            pass
        return pyarrow.dataset.dataset(self.path, format="parquet", filesystem=self.fs)


@registry.readers.register("parquet")
def read_parquet(
    path: Union[str, Path],
    converter: Optional[Union[str, Callable]] = None,
    *,
    filesystem: Optional[FileSystem] = None,
    shuffle: Literal["dataset", "file", False] = False,
    seed: Optional[int] = None,
    loop: bool = False,
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
    filesystem: Optional[AbstractFileSystem] = None,
        The filesystem to use to write the files. If None, the filesystem will be
        inferred from the path (e.g. `s3://` will use S3).
    shuffle: Literal["dataset", "file", False]
        Whether to shuffle the data. If "dataset", the whole dataset will be shuffled
        before starting iterating on it (at the start of every epoch if looping). If
        "file", shuffling will occur between and inside the parquet files, but not
        across them.

        !!! warning "Dataset shuffling"

            Shuffling the dataset can be expensive, especially for large datasets,
            since it requires reading the entire dataset into memory. If you have a
            large dataset, consider shuffling at the "file" level.

    seed: Optional[int]
        The seed to use for shuffling.
    loop: bool
        Whether to loop over the data indefinitely.
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the parquet rows of the data source to Doc objects
        These are documented on the [Converters](/data/converters) page.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.

    Returns
    -------
    LazyCollection
    """
    if "read_in_worker" in kwargs:
        warnings.warn(
            "The `read_in_worker` parameter of edsnlp.data.read_parquet is deprecated "
            "and set to True by default.",
            FutureWarning,
        )
        kwargs.pop("read_in_worker")

    data = LazyCollection(
        reader=ParquetReader(
            path,
            filesystem=filesystem,
            shuffle=shuffle,
            seed=seed,
            loop=loop,
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
    batch_size: Optional[int] = None,
    batch_by: Union[Callable, Literal["record"]] = "record",
    batch_in_worker: bool = True,
    overwrite: bool = False,
    filesystem: Optional[FileSystem] = None,
    execute: bool = True,
    converter: Optional[Union[str, Callable]] = None,
    **kwargs,
) -> pyarrow.dataset.Dataset:
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
    batch_size: Optional[int]
        The maximum number of documents to write in each parquet file.
    batch_by: Union[Callable, Literal["record", "file"]]
        The method to batch the documents. If "record", the batch size is the number of
        documents. If "file", each batch corresponds to a parquet file fragment from
        the input data.
    batch_in_worker: bool
        In multiprocessing or spark mode, whether to batch the documents in the workers
        or in the main process.

        For instance, a worker may read the 1st, 3rd, 5th, ... documents, while another
        reads the 2nd, 4th, 6th, ... documents. If `batch_in_worker` is False and
        `deterministic` is True (default), the original order of the documents will be
        recovered in the main process, and batching there can produce fragments that
        respect the original order.
    overwrite: bool
        Whether to overwrite existing directories.
    filesystem: Optional[AbstractFileSystem] = None,
        The filesystem to use to write the files. If None, the filesystem will be
        inferred from the path (e.g. `s3://` will use S3).
    execute: bool
        Whether to execute the writing operation immediately or to return a lazy
        collection
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the documents to dictionary objects before writing
        them as Parquet rows. These are documented on the [Converters](/data/converters)
        page.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.
    """

    data = LazyCollection.ensure_lazy(data)
    if "num_rows_per_file" in kwargs:
        assert (
            batch_size is None
        ), "Cannot specify both 'batch_size' and deprecated 'num_rows_per_file'."
        batch_size = kwargs.pop("num_rows_per_file")
        assert batch_by == "record", "Cannot use 'num_rows_per_file' with 'batch_by'."
    if "write_in_worker" in kwargs:
        warnings.warn(
            "The 'write_in_worker' parameter is deprecated. To perform "
            "batching in the worker processes, set 'batch_in_worker=True'.",
            VisibleDeprecationWarning,
        )
        batch_in_worker = kwargs.pop("write_in_worker")
    if "accumulate" in kwargs:
        warnings.warn(
            "The 'accumulate' parameter is deprecated.", VisibleDeprecationWarning
        )
    if converter:
        converter, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)

    return data.write(
        ParquetWriter(
            path=path,
            batch_size=batch_size,
            batch_by=batch_by,
            batch_in_worker=batch_in_worker,
            overwrite=overwrite,
            filesystem=filesystem,
        ),
        execute=execute,
    )
