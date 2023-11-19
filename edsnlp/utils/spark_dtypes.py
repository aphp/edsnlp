import contextlib
import pprint
import warnings
from array import array

from pyspark.sql.types import (
    ArrayType,
    DataType,
    DecimalType,
    NullType,
    StructField,
    StructType,
    _array_type_mappings,
    _infer_schema,
    _infer_type,
    _type_mappings,
)


class PySparkPrettyPrinter(pprint.PrettyPrinter):
    def __init__(self, *args, width=1, **kwargs):
        width = 1
        super().__init__(*args, width=width, **kwargs)

    def _format(self, object, stream, indent, allowance, context, level):
        if isinstance(object, DataType):
            self._pprint_pyspark_type(object, stream, indent, allowance, context, level)
        else:
            super()._format(object, stream, indent, allowance, context, level)

    _dispatch = pprint.PrettyPrinter._dispatch.copy()

    def _pprint_pyspark_type(self, object, stream, indent, allowance, context, level):
        if isinstance(object, StructType):
            stream.write("T.StructType([\n")
            for i, field in enumerate(object.fields):
                if i > 0:
                    stream.write(",\n")
                stream.write(" " * (indent + 4))  # Increase indent for nested elements
                self._pprint_pyspark_type(
                    field, stream, indent + 4, allowance, context, level + 1
                )
            stream.write("\n" + " " * indent + "])")
        elif isinstance(object, StructField):
            stream.write(f"T.StructField('{object.name}', ")
            self._pprint_pyspark_type(
                object.dataType, stream, indent, allowance, context, level + 1
            )
            stream.write(f", {object.nullable})")
        elif isinstance(object, ArrayType):
            stream.write("T.ArrayType(")
            self._pprint_pyspark_type(
                object.elementType, stream, indent, allowance, context, level + 1
            )
            stream.write(f", {object.containsNull})")
        else:
            stream.write(f"T.{type(object).__name__}()")

    for dtype in _type_mappings.values():
        _dispatch[dtype.__repr__] = _pprint_pyspark_type
    _dispatch[DataType.__repr__] = _pprint_pyspark_type


def schema_warning(schema):
    warnings.warn(
        "The following schema was inferred from the dataframe. You should pass "
        "it to `edsnlp.data.to_spark` (and maybe fix it) to speed up the process "
        "and avoid surprises:\n"
        "import pyspark.sql.types as T\n"
        "dtypes = " + PySparkPrettyPrinter().pformat(schema),
        Warning,
    )


def infer_type(obj, **kwargs):
    """Infer the DataType from obj"""
    if obj is None:
        return NullType()

    if hasattr(obj, "__UDT__"):
        return obj.__UDT__

    dataType = _type_mappings.get(type(obj))
    if dataType is DecimalType:
        # the precision and scale of `obj` may be different from row to row.
        return DecimalType(38, 18)
    elif dataType is not None:
        return dataType()

    # if isinstance(obj, dict):
    #     for key, value in obj.items():
    #         if key is not None and value is not None:
    #             return MapType(_infer_type(key), _infer_type(value), True)
    #     return MapType(NullType(), NullType(), True)
    # elif isinstance(obj, list):
    if isinstance(obj, list):
        for v in obj:
            if v is not None:
                return ArrayType(_infer_type(obj[0]), True)
        return ArrayType(NullType(), True)
    elif isinstance(obj, array):
        if obj.typecode in _array_type_mappings:
            return ArrayType(_array_type_mappings[obj.typecode](), False)
        else:
            raise TypeError("not supported type: array(%s)" % obj.typecode)
    else:
        try:
            return _infer_schema(obj)
        except TypeError:
            raise TypeError("not supported type: %s" % type(obj))


def infer_schema(row, names=None, **kwargs):
    """Infer the schema from dict/namedtuple/object"""
    if isinstance(row, dict):
        # PATCHED: don't sort the items.
        # items = sorted(row.items())
        items = list(row.items())

    elif isinstance(row, (tuple, list)):
        if hasattr(row, "__fields__"):  # Row
            items = zip(row.__fields__, tuple(row))
        elif hasattr(row, "_fields"):  # namedtuple
            items = zip(row._fields, tuple(row))
        else:
            if names is None:
                names = ["_%d" % i for i in range(1, len(row) + 1)]
            elif len(names) < len(row):
                names.extend("_%d" % i for i in range(len(names) + 1, len(row) + 1))
            items = zip(names, row)

    elif hasattr(row, "__dict__"):  # object
        items = list(row.__dict__.items())

    else:
        raise TypeError("Can not infer schema for type: %s" % type(row))

    fields = []
    for k, v in items:
        try:
            fields.append(StructField(k, _infer_type(v), True))
        except TypeError as e:
            raise TypeError(
                "Unable to infer the type of the field {}.".format(k)
            ) from e
    return StructType(fields)


@contextlib.contextmanager
def spark_interpret_dicts_as_rows():
    # Replace the code object inside _infer_type
    # with a code object that has the same bytecode
    old_infer_type_code = _infer_type.__code__
    old_infer_schema_code = _infer_type.__code__
    old_infer_type_defaults = _infer_type.__defaults__
    old_infer_schema_defaults = _infer_schema.__defaults__

    _infer_type.__code__ = infer_type.__code__
    _infer_schema.__code__ = infer_schema.__code__
    _infer_type.__defaults__ = infer_type.__defaults__
    _infer_schema.__defaults__ = infer_schema.__defaults__

    yield

    _infer_type.__code__ = old_infer_type_code
    _infer_schema.__code__ = old_infer_schema_code
    _infer_type.__defaults__ = old_infer_type_defaults
    _infer_schema.__defaults__ = old_infer_schema_defaults
