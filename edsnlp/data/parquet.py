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
from edsnlp.core.stream import Stream
from edsnlp.data.base import BatchWriter, FileBasedReader
from edsnlp.data.converters import get_dict2doc_converter, get_doc2dict_converter
from edsnlp.utils.batching import BatchBy, batchify_fns
from edsnlp.utils.collections import batchify, dl_to_ld, flatten, ld_to_dl, shuffle
from edsnlp.utils.file_system import FileSystem, normalize_fs_path
from edsnlp.utils.stream_sentinels import DatasetEndSentinel, FragmentEndSentinel
from edsnlp.utils.typing import AsList


class ParquetReader(FileBasedReader):
    DATA_FIELDS = ()

    def __init__(
        self,
        path: Union[str, Path],
        *,
        filesystem: Optional[FileSystem] = None,
        shuffle: Literal["dataset", "fragment", False] = False,
        seed: Optional[int] = None,
        loop: bool = False,
        work_unit: Literal["record", "fragment"] = "record",
    ):
        super().__init__()
        self.shuffle = shuffle
        self.emitted_sentinels = {"dataset"} | (
            set() if shuffle == "dataset" else {"fragment"}
        )
        seed = seed if seed is not None else random.getrandbits(32)
        self.rng = random.Random(seed)
        self.loop = loop
        self.work_unit = work_unit
        assert not (work_unit == "fragment" and shuffle == "dataset"), (
            "Cannot shuffle at the dataset level and dispatch tasks at the "
            "fragment level. Set shuffle='fragment' or work_unit='record'."
        )
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
        return (
            doc
            for batch in fragment.scanner().to_reader()
            for doc in dl_to_ld(batch.to_pydict())
        )

    def extract_task(self, item):
        if self.work_unit == "fragment":
            records = self.read_fragment(item)
            if self.shuffle == "fragment":
                records = shuffle(records, self.rng)
            yield from records
        else:
            yield item

    def read_records(self) -> Iterable[Any]:
        while True:
            files = self.fragments
            if self.shuffle:
                files = shuffle(files, self.rng)
            if self.shuffle == "fragment":
                for file in files:
                    if self.work_unit == "fragment":
                        yield file
                    else:
                        yield from shuffle(self.read_fragment(file), self.rng)
                    yield FragmentEndSentinel(file.path)
            elif self.shuffle == "dataset":
                assert self.work_unit == "record"
                records = (line for file in files for line in self.read_fragment(file))
                yield from shuffle(records, self.rng)
            else:
                for file in files:
                    if self.work_unit == "fragment":
                        yield file
                    else:
                        yield from self.read_fragment(file)
                    yield FragmentEndSentinel(file.path)
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


class ParquetWriter(BatchWriter):
    def __init__(
        self,
        *,
        path: Union[str, Path],
        batch_size: Optional[Union[int, str]] = None,
        batch_by: BatchBy = None,
        write_in_worker: bool = False,
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
        batch_size, batch_by = Stream.validate_batching(batch_size, batch_by)
        if batch_by in ("docs", "doc", None, batchify) and batch_size is None:
            warnings.warn(
                "You should specify a batch size when using record-wise batch writing. "
                "Setting batch size to 1024."
            )
            batch_size = 1024
        batch_by = batch_by or "docs"
        self.batch_fn = batchify_fns.get(batch_by, batch_by)

        self.batch_size = batch_size
        self.write_in_worker = write_in_worker
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
    converter: Optional[AsList[Union[str, Callable]]] = None,
    *,
    filesystem: Optional[FileSystem] = None,
    shuffle: Literal["dataset", "fragment", False] = False,
    seed: Optional[int] = None,
    loop: bool = False,
    work_unit: Literal["record", "fragment"] = "record",
    **kwargs,
) -> Stream:
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
        [Stream][edsnlp.core.stream.Stream].
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
    shuffle: Literal["dataset", "fragment", False]
        Whether to shuffle the data. If "dataset", the whole dataset will be shuffled
        before starting iterating on it (at the start of every epoch if looping). If
        "fragment", shuffling will occur between and inside the parquet files, but not
        across them.

        !!! warning "Dataset shuffling"

            Shuffling the dataset can be expensive, especially for large datasets,
            since it requires reading the entire dataset into memory. If you have a
            large dataset, consider shuffling at the "fragment" level.

    seed: Optional[int]
        The seed to use for shuffling.
    loop: bool
        Whether to loop over the data indefinitely.
    work_unit: Literal["record", "fragment"]
        Only affects the multiprocessing mode. If "record", every worker will start to
        read the same parquet file and yield each every num_workers-th record, starting
        at an offset each. For instance, if num_workers=2, the first worker will read
        the 1st, 3rd, 5th, ... records, while the second worker will read the 2nd, 4th,
        6th, ... records of the first parquet file.

        If "fragment", each worker will read a different parquet file. For instance, the
        first worker will every record of the 1st parquet file, the second worker will
        read every record of the 2nd parquet file, and so on. This way, no record is
        "wasted" and every record loaded in memory is yielded.

    converter: Optional[AsList[Union[str, Callable]]]
        Converters to use to convert the parquet rows of the data source to Doc objects
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
            "The `read_in_worker` parameter of edsnlp.data.read_parquet is deprecated "
            "and set to True by default.",
            FutureWarning,
        )
        kwargs.pop("read_in_worker")

    data = Stream(
        reader=ParquetReader(
            path,
            filesystem=filesystem,
            shuffle=shuffle,
            seed=seed,
            loop=loop,
            work_unit=work_unit,
        )
    )
    if converter:
        for conv in converter:
            conv, kw = get_dict2doc_converter(conv, kwargs)
            data = data.map(conv, kwargs=kw)
    return data


@registry.writers.register("parquet")
def write_parquet(
    data: Union[Any, Stream],
    path: Union[str, Path],
    *,
    batch_size: Optional[Union[int, str]] = None,
    batch_by: BatchBy = None,
    write_in_worker: bool = True,
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
    data: Union[Any, Stream],
        The data to write (either a list of documents or a Stream).
    path: Union[str, Path]
        Path to the directory containing the parquet files (will recursively look for
        files in subdirectories). Supports any filesystem supported by pyarrow.
    batch_size: Optional[int]
        The maximum number of documents to write in each parquet file.
    batch_by: Union[Callable, Literal["docs", "fragment"]]
        The method to batch the documents. If "docs", the batch size is the number of
        documents. If "fragment", each batch corresponds to a parquet file fragment from
        the input data.
    write_in_worker: bool
        In multiprocessing or spark mode, whether to batch and write the documents in
        the workers or in the main process.

        For instance, a worker may read the 1st, 3rd, 5th, ... documents, while another
        reads the 2nd, 4th, 6th, ... documents.

        If `write_in_worker` is False, `deterministic` is True (default) and no
        operation adds or remove document from the stream (e.g., no `map_batches`), the
        original order of the documents will be recovered in the main process, and
        batching there can produce fragments that respect the original order.
    overwrite: bool
        Whether to overwrite existing directories.
    filesystem: Optional[AbstractFileSystem] = None,
        The filesystem to use to write the files. If None, the filesystem will be
        inferred from the path (e.g. `s3://` will use S3).
    execute: bool
        Whether to execute the writing operation immediately or to return a stream
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the documents to dictionary objects before writing
        them as Parquet rows. These are documented on the [Converters](/data/converters)
        page.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.
    """

    data = Stream.ensure_stream(data)
    if "num_rows_per_file" in kwargs:
        assert (
            batch_size is None
        ), "Cannot specify both 'batch_size' and deprecated 'num_rows_per_file'."
        batch_size = kwargs.pop("num_rows_per_file")
        assert batch_by in (
            None,
            "docs",
        ), "Cannot use 'num_rows_per_file' with 'batch_by'."
    if "accumulate" in kwargs:
        warnings.warn(
            "The 'accumulate' parameter is deprecated.", VisibleDeprecationWarning
        )
    if converter:
        converter, kw = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kw)

    return data.write(
        ParquetWriter(
            path=path,
            batch_size=batch_size,
            batch_by=batch_by,
            write_in_worker=write_in_worker,
            overwrite=overwrite,
            filesystem=filesystem,
        ),
        execute=execute,
    )
