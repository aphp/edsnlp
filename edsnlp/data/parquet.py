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
from edsnlp.utils.collections import dl_to_ld, flatten, flatten_once, ld_to_dl


class ParquetReader(BaseReader):
    DATA_FIELDS = ("dataset",)

    def __init__(
        self,
        path: Union[str, Path],
        *,
        read_in_worker: bool,
        filesystem: Optional[pyarrow.fs.FileSystem] = None,
    ):
        super().__init__()
        # Either the filesystem has not been passed
        # or the path is a URL (e.g. s3://) => we need to infer the filesystem
        fs_path = path
        if filesystem is None or (isinstance(path, str) and "://" in path):
            path = (
                path
                if isinstance(path, Path) or "://" in path
                else f"file://{os.path.abspath(path)}"
            )
            inferred_fs, fs_path = pyarrow.fs.FileSystem.from_uri(path)
            filesystem = filesystem or inferred_fs
            assert inferred_fs.type_name == filesystem.type_name, (
                f"Protocol {inferred_fs.type_name} in path does not match "
                f"filesystem {filesystem.type_name}"
            )
        self.read_in_worker = read_in_worker
        self.dataset = pyarrow.dataset.dataset(
            fs_path, format="parquet", filesystem=filesystem
        )

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
        filesystem: Optional[pyarrow.fs.FileSystem] = None,
    ):
        super().__init__()
        fs_path = path
        if filesystem is None or (isinstance(path, str) and "://" in path):
            path = (
                path
                if isinstance(path, Path) or "://" in path
                else f"file://{os.path.abspath(path)}"
            )
            inferred_fs, fs_path = pyarrow.fs.FileSystem.from_uri(path)
            filesystem = filesystem or inferred_fs
            assert inferred_fs.type_name == filesystem.type_name, (
                f"Protocol {inferred_fs.type_name} in path does not match "
                f"filesystem {filesystem.type_name}"
            )
            path = fs_path
        # Check that filesystem has the same protocol as indicated by path
        filesystem.create_dir(fs_path, recursive=True)
        if overwrite is False:
            dataset = pyarrow.dataset.dataset(
                fs_path, format="parquet", filesystem=filesystem
            )
            if len(list(dataset.get_fragments())):
                raise FileExistsError(
                    f"Directory {fs_path} already exists and is not empty. "
                    "Use overwrite=True to overwrite."
                )
        self.filesystem = filesystem
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
                fragment = pyarrow.Table.from_pydict(ld_to_dl(flatten(self.batch)))  # type: ignore
                count += len(self.batch)
                self.batch = []
                if self.write_in_worker:
                    pyarrow.parquet.write_to_dataset(
                        table=fragment,
                        root_path=self.path,
                        filesystem=self.filesystem,
                    )
                    fragment = None
                results.append(fragment)
        return results, count

    def finalize(self):
        if not self.finalized:
            self.finalized = True
            return self.write_worker([], last=True)

    def write_main(self, fragments: Iterable[List[Union[pyarrow.Table, Path]]]):
        for table in flatten_once(fragments):
            if not self.write_in_worker:
                pyarrow.parquet.write_to_dataset(
                    table=table,
                    root_path=self.path,
                    filesystem=self.filesystem,
                )
        return pyarrow.dataset.dataset(
            self.path, format="parquet", filesystem=self.filesystem
        )


@registry.readers.register("parquet")
def read_parquet(
    path: Union[str, Path],
    converter: Union[str, Callable],
    *,
    read_in_worker: bool = False,
    filesystem: Optional[pyarrow.fs.FileSystem] = None,
    **kwargs,
) -> LazyCollection:
    data = LazyCollection(
        reader=ParquetReader(
            path,
            read_in_worker=read_in_worker,
            filesystem=filesystem,
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
    filesystem: Optional[pyarrow.fs.FileSystem] = None,
    accumulate: bool = True,
    converter: Optional[Union[str, Callable]],
    **kwargs,
) -> None:
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
            filesystem=filesystem,
        )
    )
