from __future__ import annotations

from itertools import chain
from typing import Any, Callable, Optional, Union

import pyspark.sql.dataframe

from edsnlp import registry
from edsnlp.core.lazy_collection import LazyCollection
from edsnlp.data.base import BaseReader, BaseWriter
from edsnlp.data.converters import (
    FILENAME,
    get_dict2doc_converter,
    get_doc2dict_converter,
    without_filename,
)
from edsnlp.utils.collections import flatten_once


class SparkReader(BaseReader):
    DATA_FIELDS = ("data",)

    def __init__(self, data: pyspark.sql.dataframe.DataFrame):
        import pyspark.sql.dataframe

        self.data = data
        assert isinstance(
            self.data, (pyspark.sql.dataframe.DataFrame, chain)
        ), f"`data` should be a pyspark or koalas DataFrame got {type(data)}"
        super().__init__()

    def read_main(self):
        return ((d, 1) for d in self.data.toLocalIterator())

    def read_worker(self, fragment):
        return [task.asDict(recursive=True) for task in fragment]


@registry.readers.register("spark")
def from_spark(
    data,
    converter: Union[str, Callable],
    **kwargs,
) -> LazyCollection:
    """
    The SparkReader (or `edsnlp.data.from_spark`) reads a pyspark (or koalas) DataFrame
    and yields documents. At the moment, only entities and span attributes are loaded.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)
    doc_iterator = edsnlp.data.from_spark(note_df, converter="omop")
    annotated_docs = nlp.pipe(doc_iterator)
    ```

    !!! note "Generator vs list"

        `edsnlp.data.from_spark` returns a
        [LazyCollection][edsnlp.core.lazy_collection.LazyCollection]
        To iterate over the documents multiple times efficiently or to access them by
        index, you must convert it to a list

        ```{ .python .no-check }
        docs = list(edsnlp.data.from_spark(note_df, converter="omop"))
        ```

    Parameters
    ----------
    data: pyspark.sql.dataframe.DataFrame
        The DataFrame to read.
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the rows of the DataFrame to Doc objects.
        These are documented on the [Converters](/data/converters) page.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.

    Returns
    -------
    LazyCollection
    """
    data = LazyCollection(reader=SparkReader(data))
    if converter:
        converter, kwargs = get_dict2doc_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)
    return data


class SparkWriter(BaseWriter):
    def __init__(
        self,
        *,
        dtypes: Any = None,
        show_dtypes: bool = True,
    ):
        self.dtypes = dtypes
        self.show_dtypes = show_dtypes

        super().__init__()

    def write_worker(self, records):
        # We flatten in case the converter returns a list
        records = list(flatten_once(records))
        for rec in records:
            rec.pop(FILENAME, None)
        return records, len(records)

    def write_main(self, fragments):
        import pyspark

        from edsnlp.utils.spark_dtypes import (
            schema_warning,
            spark_interpret_dicts_as_rows,
        )

        fragments = map(without_filename, flatten_once(fragments))
        spark = pyspark.sql.SparkSession.builder.enableHiveSupport().getOrCreate()
        rdd = (
            fragments
            if isinstance(fragments, pyspark.RDD)
            else spark.sparkContext.parallelize(fragments)
        )
        with spark_interpret_dicts_as_rows():
            result = spark.createDataFrame(rdd, schema=self.dtypes)

        if self.dtypes is None and self.show_dtypes:
            schema_warning(result.schema)

        return result


@registry.writers.register("spark")
def to_spark(
    data: Union[Any, LazyCollection],
    converter: Optional[Union[str, Callable]] = None,
    dtypes: Any = None,
    show_dtypes: bool = True,
    **kwargs,
):
    """
    `edsnlp.data.to_spark` converts a list of documents into a Spark DataFrame, usually
    one row per document, unless the converter returns a list in which case each entry
    of the resulting list will be stored in its own row.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.covid")

    note_df = sql('''
        select note_id, note_text from note
        where note_text is not null
        limit 500
    ''')

    docs = edsnlp.data.from_spark(note_df, converter="omop")

    docs = nlp.pipe(docs)

    res = edsnlp.data.to_spark(docs, converter="omop")

    res.show()
    ```

    !!! tip "Mac OS X"

        If you are using Mac OS X, you may need to set the following environment
        variable (see [this thread](https://stackoverflow.com/a/52230415)) to run
        pyspark:

        ```{ .python .no-check }
        import os
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        ```

    Parameters
    ----------
    data: Union[Any, LazyCollection],
        The data to write (either a list of documents or a LazyCollection).
    dtypes: pyspark.sql.types.StructType
        The schema to use for the DataFrame.
    show_dtypes: bool
        Whether to print the inferred schema (only if `dtypes` is None).
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

    return data.write(SparkWriter(dtypes=dtypes, show_dtypes=show_dtypes))
