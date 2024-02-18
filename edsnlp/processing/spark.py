from __future__ import annotations

import pickle
import sys
from typing import Optional

import dill

from edsnlp.core.lazy_collection import LazyCollection
from edsnlp.data.base import BaseWriter
from edsnlp.data.converters import set_current_tokenizer
from edsnlp.data.spark import SparkReader, SparkWriter
from edsnlp.utils.collections import batchify, flatten_once

try:
    from koalas.dataframe import DataFrame as KoalasDataFrame
except ImportError:  # pragma: no cover
    KoalasDataFrame = None


class Broadcasted:
    def __init__(self, value):
        self.value = value

    def __getstate__(self):
        return dill.dumps(self.value, byref=True, recurse=True)

    def __setstate__(self, state):  # pragma: no cover
        self.value = dill.loads(state)


def execute_spark_backend(
    lc: LazyCollection,
):
    """
    This execution mode uses Spark to parallelize the processing of the documents.
    The documents are first stored in a Spark DataFrame (if it was not already the case)
    and then processed in parallel using Spark.

    Beware, if the original reader was not a SparkReader (`edsnlp.data.from_spark`), the
    *local docs* → *spark dataframe* conversion might take some time, and the whole
    process might be slower than using the `multiprocessing` backend.
    """
    import pyspark.sql.types as T
    from pyspark.sql import SparkSession

    from edsnlp.utils.spark_dtypes import schema_warning, spark_interpret_dicts_as_rows

    try:
        getActiveSession = SparkSession.getActiveSession
    except AttributeError:

        def getActiveSession() -> Optional["SparkSession"]:  # pragma: no cover
            from pyspark import SparkContext

            sc = SparkContext._active_spark_context
            if sc is None:
                return None
            else:
                assert sc._jvm is not None
                if sc._jvm.SparkSession.getActiveSession().isDefined():
                    SparkSession(sc, sc._jvm.SparkSession.getActiveSession().get())
                    try:
                        return SparkSession._activeSession
                    except AttributeError:
                        try:
                            return SparkSession._instantiatedSession
                        except AttributeError:
                            return None
                else:
                    return None

    # Get current spark session
    spark = getActiveSession() or SparkSession.builder.getOrCreate()

    reader = lc.reader
    writer = lc.writer

    if (
        writer is not None
        # check if finalize has been overridden
        and writer.finalize.__func__ is not BaseWriter.finalize
    ):
        raise ValueError(
            "The Spark backend does not support writers that need to be finalized."
        )

    if isinstance(lc.reader, SparkReader):
        df = lc.reader.data
    else:
        with spark_interpret_dicts_as_rows():
            df = spark.sparkContext.parallelize(
                [
                    {"content": pickle.dumps(item, -1)}
                    for item, count in reader.read_main()
                ]
            ).toDF(T.StructType([T.StructField("content", T.BinaryType())]))

    def process_partition(iterator):  # pragma: no cover
        lc: LazyCollection = bc.value
        try:
            sys.modules["torch"].set_grad_enabled(False)
        except (AttributeError, KeyError):
            pass

        results = []
        tasks = (
            # Maybe we should skip read_worker here ? it's a no-op for SparkReader
            lc.reader.read_worker(iterator)
            if isinstance(lc.reader, SparkReader)
            else (
                task
                for row in iterator
                for task in lc.reader.read_worker([pickle.loads(row["content"])])
            )
        )
        for batch in batchify(
            tasks,
            batch_size=lc.batch_size,
        ):
            with lc.cache():
                for name, pipe, kwargs, tokenizer in lc.pipeline:
                    with set_current_tokenizer(tokenizer):
                        if hasattr(pipe, "batch_process"):
                            batch = pipe.batch_process(batch, **kwargs)
                        else:
                            batch = [pipe(doc, **kwargs) for doc in batch]  # type: ignore
            results.extend(batch)

        if lc.writer:
            results, count = lc.writer.write_worker(results)

        if isinstance(lc.writer, SparkWriter):
            return results
        else:
            return [{"content": pickle.dumps(results, -1)}]

    with lc.eval():
        bc = Broadcasted(lc.worker_copy())

        if isinstance(writer, SparkWriter):
            rdd = df.rdd.mapPartitions(process_partition)
            with spark_interpret_dicts_as_rows():
                results = spark.createDataFrame(rdd, schema=writer.dtypes)

            if writer.dtypes is None and writer.show_dtypes:
                schema_warning(results.schema)
            return results

        results = (
            pickle.loads(item["content"])
            for item in df.rdd.mapPartitions(process_partition).toLocalIterator()
        )
        return (
            writer.write_main(results) if writer is not None else flatten_once(results)
        )
