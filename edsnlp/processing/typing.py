import importlib
from enum import Enum
from typing import Union


class DataFrameModules(Enum):
    PANDAS = "pandas"
    PYSPARK = "pyspark.sql"
    KOALAS = "databricks.koalas"


DataFrames = None

for module in list(DataFrameModules):
    try:
        spec = importlib.util.find_spec(module.value)
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        DataFrames = Union[DataFrames, importlib.import_module(module.value).DataFrame]


def get_module(df: DataFrames):
    for module in list(DataFrameModules):
        if df.__class__.__module__.startswith(module.value):
            return module
