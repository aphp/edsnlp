from typing import TYPE_CHECKING
from edsnlp.utils.lazy_module import lazify

lazify()

if TYPE_CHECKING:
    from .base import from_iterable, to_iterable
    from .standoff import read_standoff, write_standoff
    from .brat import read_brat, write_brat
    from .conll import read_conll
    from .json import read_json, write_json
    from .parquet import read_parquet, write_parquet
    from .spark import from_spark, to_spark
    from .pandas import from_pandas, to_pandas
    from .polars import from_polars, to_polars
    from .converters import get_dict2doc_converter, get_doc2dict_converter
