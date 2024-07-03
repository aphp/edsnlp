import datetime
from array import array
from collections import namedtuple
from decimal import Decimal

import numpy as np
import pytest

mytype = namedtuple("mytype", ["a", "b", "c", "z"])


def test_infer_schema():
    from pyspark import Row
    from pyspark.sql.types import StringType

    from edsnlp.utils.spark_dtypes import (
        PySparkPrettyPrinter,
        infer_schema,
        schema_warning,
        spark_interpret_dicts_as_rows,
    )

    class CustomObj:
        def __init__(self, value):
            self.value = value

    class HasUDT:
        __UDT__ = StringType()

        def __init__(self, value: str):
            self.value = value

    with spark_interpret_dicts_as_rows():
        result = infer_schema(
            {
                "null": None,
                "null_list": [None, None, None],
                "a": 1.0,
                "b": datetime.datetime(2021, 1, 1),
                "c": [
                    {
                        "d": "foo",
                        "e": 4,
                    },
                    {
                        "d": "bar",
                        "e": 6,
                    },
                ],
                "z": Decimal("1.0"),
                "struct": [mytype(1, 2, 3, 4), mytype(5, 6, 7, 8)],
                "custom": CustomObj(1),
                "udt": HasUDT("foo"),
                "rows": [Row(a=1, b=2), Row(a=3, b=4)],
                "tuples": [(1, 2), (3, 4)],
            }
        )
        assert PySparkPrettyPrinter().pformat(result) == (
            "T.StructType([\n"
            "    T.StructField('null', T.NullType(), True),\n"
            "    T.StructField('null_list', T.ArrayType(T.NullType(), True), True),\n"
            "    T.StructField('a', T.DoubleType(), True),\n"
            "    T.StructField('b', T.TimestampType(), True),\n"
            "    T.StructField('c', T.ArrayType(T.StructType([\n"
            "        T.StructField('d', T.StringType(), True),\n"
            "        T.StructField('e', T.LongType(), True)\n"
            "    ]), True), True),\n"
            "    T.StructField('z', T.DecimalType(), True),\n"
            "    T.StructField('struct', T.ArrayType(T.StructType([\n"
            "        T.StructField('a', T.LongType(), True),\n"
            "        T.StructField('b', T.LongType(), True),\n"
            "        T.StructField('c', T.LongType(), True),\n"
            "        T.StructField('z', T.LongType(), True)\n"
            "    ]), True), True),\n"
            "    T.StructField('custom', T.StructType([\n"
            "        T.StructField('value', T.LongType(), True)\n"
            "    ]), True),\n"
            "    T.StructField('udt', T.StringType(), True),\n"
            "    T.StructField('rows', T.ArrayType(T.StructType([\n"
            "        T.StructField('a', T.LongType(), True),\n"
            "        T.StructField('b', T.LongType(), True)\n"
            "    ]), True), True),\n"
            "    T.StructField('tuples', T.ArrayType(T.StructType([\n"
            "        T.StructField('_1', T.LongType(), True),\n"
            "        T.StructField('_2', T.LongType(), True)\n"
            "    ]), True), True)\n"
            "])"
        )

        with pytest.raises(TypeError):
            infer_schema({"numpy": np.array([1, 2, 3])})

        result = infer_schema(
            (1, 2, 3, 4, 5),
            names=["a", "b", "c"],
        )
        assert PySparkPrettyPrinter().pformat(result) == (
            "T.StructType([\n"
            "    T.StructField('a', T.LongType(), True),\n"
            "    T.StructField('b', T.LongType(), True),\n"
            "    T.StructField('c', T.LongType(), True),\n"
            "    T.StructField('_4', T.LongType(), True),\n"
            "    T.StructField('_5', T.LongType(), True)\n"
            "])"
        )

        result = infer_schema(
            {"field": array("d", [1.0, 2.0, 3.14])},
        )
        assert PySparkPrettyPrinter().pformat({"type": result}) == (
            "{'type': T.StructType([\n"
            "             T.StructField('field', T.ArrayType(T.DoubleType(), False), "
            "True)\n"
            "         ])}"
        )

        with pytest.raises(TypeError):
            infer_schema({"error_array": array("L", [1, 2, 3])})

        with pytest.warns(Warning) as warned:
            schema_warning(result)

        assert len(warned) == 1
        assert "The following schema was inferred" in warned[0].message.args[0]
