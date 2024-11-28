from __future__ import annotations

import random
from itertools import chain
from typing import Any, Callable, Iterable, Optional, Union

import pyspark.sql.dataframe
from typing_extensions import Literal

from edsnlp import registry
from edsnlp.core.stream import Stream
from edsnlp.data.base import BaseWriter, MemoryBasedReader
from edsnlp.data.converters import (
    get_dict2doc_converter,
    get_doc2dict_converter,
    without_filename,
)
from edsnlp.utils.collections import flatten
from edsnlp.utils.spark_dtypes import (
    schema_warning,
    spark_interpret_dicts_as_rows,
)
from edsnlp.utils.stream_sentinels import DatasetEndSentinel
from edsnlp.utils.typing import AsList


class SparkReader(MemoryBasedReader):
    DATA_FIELDS = ("data",)

    def __init__(
        self,
        data: pyspark.sql.dataframe.DataFrame,
        shuffle: Literal["dataset", False] = False,
        seed: Optional[int] = None,
        loop: bool = False,
    ):
        import pyspark.sql.dataframe

        self.data = data
        self.shuffle = shuffle
        self.emitted_sentinels = {"dataset"}
        seed = seed if seed is not None else random.getrandbits(32)
        self.rng = random.Random(seed)
        self.loop = loop
        assert isinstance(
            self.data, (pyspark.sql.dataframe.DataFrame, chain)
        ), f"`data` should be a pyspark or koalas DataFrame got {type(data)}"
        super().__init__()

    def read_records(self) -> Iterable[Any]:
        while True:
            data: "pyspark.sql.dataframe.DataFrame" = self.data
            if self.shuffle == "dataset":
                data = data.sample(fraction=1.0, seed=self.rng.getrandbits(32))
            items = (item.asDict(recursive=True) for item in data.toLocalIterator())
            yield from items
            yield DatasetEndSentinel()
            if not self.loop:
                break

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={object.__repr__(self.data)}, "
            f"shuffle={self.shuffle}, "
            f"loop={self.loop})"
        )


@registry.readers.register("spark")
def from_spark(
    data,
    converter: Optional[AsList[Union[str, Callable]]] = None,
    shuffle: Literal["dataset", False] = False,
    seed: Optional[int] = None,
    loop: bool = False,
    **kwargs,
) -> Stream:
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
        [Stream][edsnlp.core.stream.Stream]
        To iterate over the documents multiple times efficiently or to access them by
        index, you must convert it to a list

        ```{ .python .no-check }
        docs = list(edsnlp.data.from_spark(note_df, converter="omop"))
        ```

    Parameters
    ----------
    data: pyspark.sql.dataframe.DataFrame
        The DataFrame to read.
    shuffle: Literal["dataset", False]
        Whether to shuffle the data. If "dataset", the whole dataset will be shuffled
        before starting iterating on it (at the start of every epoch if looping).
    seed: Optional[int]
        The seed to use for shuffling.
    loop: bool
        Whether to loop over the data indefinitely.
    converter: Optional[AsList[Union[str, Callable]]]
        Converters to use to convert the rows of the DataFrame to Doc objects.
        These are documented on the [Converters](/data/converters) page.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.

    Returns
    -------
    Stream
    """
    data = Stream(
        reader=SparkReader(
            data,
            shuffle=shuffle,
            seed=seed,
            loop=loop,
        )
    )
    if converter:
        for conv in converter:
            conv, kwargs = get_dict2doc_converter(conv, kwargs)
            data = data.map(conv, kwargs=kwargs)
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

    def consolidate(self, items: Iterable[Any]):
        items = map(without_filename, flatten(items))
        spark = pyspark.sql.SparkSession.builder.enableHiveSupport().getOrCreate()
        rdd = (
            items
            if isinstance(items, pyspark.RDD)
            else spark.sparkContext.parallelize(items)
        )
        with spark_interpret_dicts_as_rows():
            result = spark.createDataFrame(rdd, schema=self.dtypes)

        if self.dtypes is None and self.show_dtypes:
            schema_warning(result.schema)

        return result


@registry.writers.register("spark")
def to_spark(
    data: Union[Any, Stream],
    converter: Optional[Union[str, Callable]] = None,
    dtypes: Any = None,
    show_dtypes: bool = True,
    execute: bool = True,
    **kwargs,
):
    """
    `edsnlp.data.to_spark` converts a list of documents into a Spark DataFrame, usually
    one row per document, unless the converter returns a list in which case each entry
    of the resulting list will be stored in its own row.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.covid())

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
    data: Union[Any, Stream],
        The data to write (either a list of documents or a Stream).
    dtypes: pyspark.sql.types.StructType
        The schema to use for the DataFrame.
    show_dtypes: bool
        Whether to print the inferred schema (only if `dtypes` is None).
    execute: bool
        Whether to execute the writing operation immediately or to return a stream
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the documents to dictionary objects before storing
        them in the dataframe. These are documented on the
        [Converters](/data/converters) page.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.
    """
    data = Stream.ensure_stream(data)
    if converter:
        converter, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)

    return data.write(
        SparkWriter(dtypes=dtypes, show_dtypes=show_dtypes), execute=execute
    )
