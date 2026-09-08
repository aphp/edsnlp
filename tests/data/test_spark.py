import pytest

import edsnlp

pytestmark = pytest.mark.processing


def test_read_write(blank_nlp, text, df_notes_pyspark):
    # line below is just to mix params to avoid running too many tests
    shuffle = "dataset" if blank_nlp.lang == "eds" else False

    reader = edsnlp.data.from_spark(
        df_notes_pyspark,
        converter="omop",
        nlp=blank_nlp,
        shuffle=shuffle,
    ).set_processing(backend="simple")
    doc = list(reader)[0]
    assert doc.text == text

    blank_nlp.add_pipe("eds.matcher", config={"terms": {"douleur": ["douleurs"]}})
    blank_nlp.add_pipe("eds.negation")
    docs = blank_nlp.pipe(reader)

    writer = edsnlp.data.to_spark(
        docs,
        converter="omop",
        span_attributes=["negation"],
        span_getter=["ents"],
    )
    res = writer.toPandas().to_dict(orient="records")
    assert len(res) == 20
    assert sum(len(r["entities"]) for r in res) == 20


def test_spark_schema():
    from confit import VisibleDeprecationWarning
    from pyspark.sql import SparkSession
    from pyspark.sql import types as T

    spark = SparkSession.builder.master("local[1]").getOrCreate()
    schema = T.StructType(
        [T.StructField("id", T.LongType(), False, {"source": "test"})]
    )
    rows = [{"id": 1, "text": "hello"}]
    result = edsnlp.data.to_spark(
        rows, schema=schema, schema_overrides={"id": T.IntegerType()}, show_dtypes=False
    )
    assert result.schema == T.StructType(
        [T.StructField("id", T.IntegerType(), False, {"source": "test"})]
    )
    assert [row.asDict() for row in result.collect()] == [{"id": 1}]
    result = (
        edsnlp.data.from_spark(spark.createDataFrame(rows))
        .set_processing(backend="spark")
        .to_spark(
            schema=["text", "id"],
            schema_overrides={"id": T.IntegerType()},
            show_dtypes=False,
        )
    )
    assert result.columns == ["text", "id"]
    assert result.schema["id"].dataType == T.IntegerType()
    assert [row.asDict() for row in result.collect()] == rows
    result = edsnlp.data.to_spark([], schema="id INT", show_dtypes=False)
    assert result.columns == ["id"] and result.count() == 0
    with pytest.warns(VisibleDeprecationWarning):
        result = edsnlp.data.to_spark(
            rows, dtypes=["renamed", "text"], show_dtypes=False
        )
    assert result.first().renamed == 1


def test_dataframe_schema():
    from confit import VisibleDeprecationWarning
    from pyspark.sql import SparkSession
    from pyspark.sql import types as T

    SparkSession.builder.master("local[1]").getOrCreate()

    rows = [{"text": "hello", "id": 1}]
    frame = edsnlp.data.to_spark(
        rows, schema_overrides={"id": T.IntegerType()}, execute=False
    ).execute()
    assert list(frame.columns) == ["text", "id"]
    assert frame.schema["id"].dataType == T.IntegerType()

    frame = edsnlp.data.to_spark(
        [{"id": 1, "text": None, "discarded": None}, rows[0]],
        schema=["id", "text"],
    )
    assert list(frame.columns) == ["id", "text"]
    assert frame.schema["text"].dataType == T.StringType()
    for data in (rows, [], [{"id": None}]):
        frame = edsnlp.data.to_spark(
            data,
            schema=["missing", "id"],
            schema_overrides={"missing": T.IntegerType(), "id": T.IntegerType()},
        )
        assert list(frame.columns) == ["missing", "id"]
        assert [row.asDict() for row in frame.collect()] == [
            {"missing": None, "id": row.get("id")} for row in data
        ]
    frame = edsnlp.data.to_spark([], schema={"id": T.IntegerType()})
    assert list(frame.columns) == ["id"]
    assert frame.schema["id"].dataType == T.IntegerType()
    with pytest.warns(VisibleDeprecationWarning):
        frame = edsnlp.data.to_spark(
            rows, dtypes=T.StructType([T.StructField("id", T.IntegerType())])
        )
    assert list(frame.columns) == ["id"]
