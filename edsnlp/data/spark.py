from __future__ import annotations

import random
from itertools import chain
from typing import Any, Callable, Iterable, Optional, Union

import pyspark.sql.dataframe
import pyspark.sql.types as T
from typing_extensions import Literal

from edsnlp import registry
from edsnlp.core.stream import Stream
from edsnlp.data.base import (
    BaseWriter,
    MemoryBasedReader,
    validate_schema,
    validate_schema_overrides,
)
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
        assert isinstance(self.data, (pyspark.sql.dataframe.DataFrame, chain)), (
            f"`data` should be a pyspark or koalas DataFrame got {type(data)}"
        )
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
        schema: Any = None,
        schema_overrides: Optional[dict] = None,
    ):
        validate_schema(schema, schema_overrides, dtypes)
        self.dtypes = dtypes
        self.show_dtypes = show_dtypes
        self.schema = schema
        self.schema_overrides = schema_overrides or {}

        super().__init__()

    def consolidate(self, items: Iterable[Any]):
        spark = pyspark.sql.SparkSession.builder.enableHiveSupport().getOrCreate()
        rdd = (
            items
            if isinstance(items, pyspark.RDD)
            else spark.sparkContext.parallelize(map(without_filename, flatten(items)))
        )
        schema = self.dtypes
        inferred = schema is None
        with spark_interpret_dicts_as_rows():
            # PySpark doesn't support inference with a partial schema, so we replicate
            # the dtype inference mechanism to combine schema overrides with the
            # schema inferred from the data
            if self.schema is not None or self.schema_overrides:
                schema = self.schema
                if schema is None:
                    schema = list(
                        dict.fromkeys(name for row in rdd.take(100) for name in row)
                    )
                if isinstance(schema, str):
                    schema = T._parse_datatype_string(schema)
                inferred = isinstance(schema, list)
                types = dict(self.schema_overrides)
                if inferred:
                    names = [name for name in schema if name not in types]
                    # Columns with a dtype override must bypass inference, since
                    # Spark cannot infer a dtype for a column containing only None
                    if names:
                        inferred_schema = spark._inferSchema(
                            rdd.map(lambda row: {name: row.get(name) for name in names})
                        )
                        types.update(
                            {field.name: field.dataType for field in inferred_schema}
                        )
                    inferred = self.schema is None or bool(names)
                    schema = {name: types[name] for name in schema}
                if isinstance(schema, dict):
                    schema = T.StructType(
                        [T.StructField(name, dtype) for name, dtype in schema.items()]
                    )
                validate_schema_overrides(schema.names, self.schema_overrides)
                fields = [
                    T.StructField(
                        field.name,
                        types.get(field.name, field.dataType),
                        field.nullable,
                        field.metadata,
                    )
                    for field in schema
                ]
                schema = T.StructType(fields) if fields else None
            result = spark.createDataFrame(rdd, schema=schema)

        if inferred and self.show_dtypes:
            schema_warning(result.schema)

        return result


@registry.writers.register("spark")
def to_spark(
    data: Union[Any, Stream],
    converter: Optional[Union[str, Callable]] = None,
    dtypes: Any = None,
    show_dtypes: bool = True,
    execute: bool = True,
    *,
    schema: Optional[Union[list[str], dict, T.StructType, str]] = None,
    schema_overrides: Optional[dict] = None,
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
    dtypes: Any
        Deprecated, use schema instead
    schema: Optional[Union[list[str], dict]]
        Column names to keep or a mapping from column names to Spark dtypes
        Spark StructType and DDL schemas are also accepted
    schema_overrides: Optional[dict]
        Column dtypes to change without filtering columns, taking precedence
        over schema
    show_dtypes: bool
        Whether to print the schema when column names or dtypes were inferred
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
        SparkWriter(
            dtypes=dtypes,
            show_dtypes=show_dtypes,
            schema=schema,
            schema_overrides=schema_overrides,
        ),
        execute=execute,
    )
