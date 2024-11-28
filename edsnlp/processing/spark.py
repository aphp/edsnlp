from __future__ import annotations

import pickle
import sys
from typing import Optional, Union

import dill

from edsnlp.core.stream import Stream
from edsnlp.data.base import BaseWriter, BatchWriter
from edsnlp.data.spark import SparkReader, SparkWriter
from edsnlp.utils.collections import flatten, flatten_once
from edsnlp.utils.stream_sentinels import StreamSentinel

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
    stream: Stream,
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

    reader = stream.reader
    writer: Union[BaseWriter, BatchWriter] = stream.writer

    if isinstance(reader, SparkReader):
        df = reader.data
        assert not reader.loop, "Looping is not supported with Spark backend."
        df = (
            df.sample(fraction=1.0, seed=reader.rng.getrandbits(32))
            if reader.shuffle
            else df
        )
    else:
        with spark_interpret_dicts_as_rows():
            df = spark.sparkContext.parallelize(
                [
                    {"content": pickle.dumps(item, -1)}
                    for item in reader.read_records()
                    if not isinstance(item, StreamSentinel)
                ]
            ).toDF(T.StructType([T.StructField("content", T.BinaryType())]))

    def make_torch_pipe(torch_pipe, disable_after):  # pragma: no cover
        def wrapped(batches):
            for batch in batches:
                batch_id = hash(tuple(id(x) for x in batch))
                torch_pipe.enable_cache(batch_id)
                batch = torch_pipe.batch_process(batch)
                if disable_after:
                    torch_pipe.disable_cache(batch_id)
                yield batch

        return wrapped

    def process_partition(items):  # pragma: no cover
        stream: Stream = bc.value
        writer = stream.writer
        try:
            sys.modules["torch"].set_grad_enabled(False)
        except (AttributeError, KeyError):
            pass
        stages = stream._make_stages(split_torch_pipes=True)

        if not isinstance(stream.reader, SparkReader):
            items = (pickle.loads(row.content) for row in items)
        else:
            items = (item.asDict(recursive=True) for item in items)

        items = (task for item in items for task in stream.reader.extract_task(item))

        for stage_idx, stage in enumerate(stages):
            for op in stage.cpu_ops:
                items = op(items)

            if stage.gpu_op is not None:
                pipe = make_torch_pipe(stage.gpu_op, stage_idx == len(stages) - 2)
                items = pipe(items)

        if writer is not None:
            items = (writer.handle_record(item) for item in items)

        results = []
        if getattr(writer, "write_in_worker", None) is True:
            items = writer.batch_fn(items, writer.batch_size)
            # get the 1st element (2nd is the count)
            for item in items:
                item, count = writer.handle_batch(item)
                results.append(item)
        else:
            results = list(items)

        if isinstance(writer, SparkWriter):
            return list(flatten(results))
        else:
            return [{"content": pickle.dumps(results, -1)}]

    with stream.eval():
        bc = Broadcasted(stream.worker_copy())

        if isinstance(writer, SparkWriter):
            rdd = df.rdd.mapPartitions(process_partition)
            with spark_interpret_dicts_as_rows():
                results = spark.createDataFrame(rdd, schema=writer.dtypes)

            if writer.dtypes is None and writer.show_dtypes:
                schema_warning(results.schema)
            return results

        items = flatten_once(
            pickle.loads(item["content"])
            for item in df.rdd.mapPartitions(process_partition).toLocalIterator()
        )

        if getattr(writer, "write_in_worker", None) is False:
            writer: BatchWriter
            items = writer.batch_fn(items, writer.batch_size)
            # get the 1st element (2nd is the count)
            items = (writer.handle_batch(b)[0] for b in items)
        return items if writer is None else writer.consolidate(items)
