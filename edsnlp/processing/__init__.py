from typing import TYPE_CHECKING

from edsnlp.utils.lazy_module import lazify

lazify()

if TYPE_CHECKING:
    from .deprecated_pipe import pipe  # DEPRECATED
    from .spark import execute_spark_backend
    from .simple import execute_simple_backend
    from .multiprocessing import execute_multiprocessing_backend
