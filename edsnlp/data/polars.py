from __future__ import annotations

import random
from typing import Any, Callable, Iterable, Optional, Union

import polars
import polars as pl
from typing_extensions import Literal

from edsnlp import registry
from edsnlp.core.lazy_collection import LazyCollection
from edsnlp.data.base import BaseWriter, MemoryBasedReader
from edsnlp.data.converters import (
    get_dict2doc_converter,
    get_doc2dict_converter,
)
from edsnlp.utils.collections import flatten


class PolarsReader(MemoryBasedReader):
    DATA_FIELDS = ("data",)

    def __init__(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame],
        shuffle: Literal["dataset", False] = False,
        seed: Optional[int] = None,
        loop: bool = False,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.loop = loop

        if hasattr(data, "collect"):
            data = data.collect()
        assert isinstance(data, pl.DataFrame)
        self.data = data

    def read_records(self, work_unit: str = "record") -> Iterable[Any]:
        data: polars.DataFrame = self.data
        while True:
            if self.shuffle:
                data = self.data.sample(
                    fraction=1.0,
                    seed=self.rng.getrandbits(32),
                    shuffle=True,
                )
            yield from data.iter_rows(named=True)
            if not self.loop:
                break

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={object.__repr__(self.data)}, "
            f"shuffle={self.shuffle}, "
            f"loop={self.loop})"
        )


@registry.readers.register("polars")
def from_polars(
    data: Union[pl.DataFrame, pl.LazyFrame],
    shuffle: Literal["dataset", False] = False,
    seed: Optional[int] = None,
    loop: bool = False,
    converter: Optional[Union[str, Callable]] = None,
    **kwargs,
) -> LazyCollection:
    """
    The PolarsReader (or `edsnlp.data.from_polars`) handles reading from a table and
    yields documents. At the moment, only entities and attributes are loaded. Relations
    and events are not supported.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)
    doc_iterator = edsnlp.data.from_polars(df, nlp=nlp, converter="omop")
    annotated_docs = nlp.pipe(doc_iterator)
    ```

    !!! note "Generator vs list"

        `edsnlp.data.from_polars` returns a
        [LazyCollection][edsnlp.core.lazy_collection.LazyCollection].
        To iterate over the documents multiple times efficiently or to access them by
        index, you must convert it to a list

        ```{ .python .no-check }
        docs = list(edsnlp.data.from_polars(df, converter="omop"))
        ```

    Parameters
    ----------
    data: Union[pl.DataFrame, pl.LazyFrame]
        Polars object
    shuffle: Literal["dataset", False]
        Whether to shuffle the data. If "dataset", the whole dataset will be shuffled
        at the beginning (of every epoch if looping).
    seed: int
        The seed to use for shuffling.
    loop: bool
        Whether to loop over the data indefinitely.
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the rows of the DataFrame (represented as dicts)
        to Doc objects. These are documented on the [Converters](/data/converters) page.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.

    Returns
    -------
    LazyCollection
    """

    data = LazyCollection(
        reader=PolarsReader(
            data,
            shuffle=shuffle,
            seed=seed,
            loop=loop,
        )
    )
    if converter:
        converter, kwargs = get_dict2doc_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)
    return data


class PolarsWriter(BaseWriter):
    def __init__(self, dtypes: Optional[dict] = None):
        self.dtypes = dtypes

    def consolidate(self, items: Iterable[Any]):
        return pl.from_dicts(flatten(items), schema=self.dtypes)


@registry.writers.register("polars")
def to_polars(
    data: Union[Any, LazyCollection],
    converter: Optional[Union[str, Callable]] = None,
    dtypes: Optional[dict] = None,
    execute: bool = True,
    **kwargs,
) -> pl.DataFrame:
    """
    `edsnlp.data.to_polars` writes a list of documents as a polars dataframe.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)

    doc = nlp("My document with entities")

    edsnlp.data.to_polars([doc], converter="omop")
    ```

    Parameters
    ----------
    data: Union[Any, LazyCollection],
        The data to write (either a list of documents or a LazyCollection).
    dtypes: Optional[dict]
        Dictionary of column names to dtypes. This is passed to the schema parameter of
        `pl.from_dicts`.
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the documents to dictionary objects before storing
        them in the dataframe. These are documented on the
        [Converters](/data/converters) page.
    execute: bool
        Whether to execute the writing operation immediately or to return a lazy
        collection
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.
    """
    data = LazyCollection.ensure_lazy(data)
    if converter:
        converter, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)

    return data.write(PolarsWriter(dtypes), execute=execute)
