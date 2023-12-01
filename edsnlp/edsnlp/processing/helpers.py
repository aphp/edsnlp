import importlib
from distutils.version import LooseVersion
from enum import Enum
from typing import Union

from pkg_resources import VersionConflict


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


def check_spacy_version_for_context():  # pragma: no cover
    import spacy

    spacy_version = getattr(spacy, "__version__")
    if LooseVersion(spacy_version) < LooseVersion("3.2"):
        raise VersionConflict(
            "You provided a `context` argument, which only work with spacy>=3.2.\n"
            f"However, we found SpaCy version {spacy_version}.\n",
            "Please upgrade SpaCy ;)",
        )


def slugify(chained_attr: str) -> str:
    """
    Slugify a chained attribute name

    Parameters
    ----------
    chained_attr : str
        The string to slugify (replace dots by _)

    Returns
    -------
    str
        The slugified string
    """
    return chained_attr.replace(".", "_")
