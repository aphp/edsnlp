from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Tuple, Union

import polars as pl

from edsnlp import registry
from edsnlp.core.lazy_collection import LazyCollection
from edsnlp.data.base import BaseReader, BaseWriter
from edsnlp.data.converters import (
    FILENAME,
    get_dict2doc_converter,
    get_doc2dict_converter,
)
from edsnlp.utils.collections import flatten


class PolarsReader(BaseReader):
    DATA_FIELDS = ("data",)

    def __init__(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame],
        **kwargs,
    ):
        if hasattr(data, "collect"):
            data = data.collect()
        assert isinstance(data, pl.DataFrame)
        self.data = data

        super().__init__(**kwargs)

    def read_main(self) -> Iterable[Tuple[Any, int]]:
        return ((item, 1) for item in self.data.iter_rows(named=True))

    def read_worker(self, fragments):
        return [task for task in fragments]


@registry.readers.register("polars")
def from_polars(
    data,
    converter: Union[str, Callable],
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
    data: pl.DataFrame
        Polars object
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

    data = LazyCollection(reader=PolarsReader(data))
    if converter:
        converter, kwargs = get_dict2doc_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)
    return data


class PolarsWriter(BaseWriter):
    def __init__(self, dtypes: Optional[dict] = None):
        self.dtypes = dtypes

    def write_worker(self, records):
        # If write as jsonl, we will perform the actual writing in the `write` method
        for rec in records:
            if isinstance(rec, dict):
                rec.pop(FILENAME, None)
        return records, len(records)

    def write_main(self, fragments):
        return pl.from_dicts(flatten(fragments), schema=self.dtypes)


@registry.writers.register("polars")
def to_polars(
    data: Union[Any, LazyCollection],
    converter: Optional[Union[str, Callable]] = None,
    dtypes: Optional[dict] = None,
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
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.
    """
    data = LazyCollection.ensure_lazy(data)
    if converter:
        converter, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)

    return data.write(PolarsWriter(dtypes))
