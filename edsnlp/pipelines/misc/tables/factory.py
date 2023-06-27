from typing import List, Optional

from spacy.language import Language

from edsnlp.pipelines.misc.tables import TablesMatcher
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    tables_pattern=None,
    sep_pattern=None,
    attr="TEXT",
    ignore_excluded=True,
    col_names=False,
    row_names=False,
)


@deprecated_factory("tables", "eds.tables", default_config=DEFAULT_CONFIG)
@Language.factory("eds.tables", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    tables_pattern: Optional[List[str]],
    sep_pattern: Optional[List[str]],
    attr: str,
    ignore_excluded: bool,
    col_names: Optional[bool] = False,
    row_names: Optional[bool] = False,
):
    return TablesMatcher(
        nlp,
        tables_pattern=tables_pattern,
        sep_pattern=sep_pattern,
        attr=attr,
        ignore_excluded=ignore_excluded,
        col_names=col_names,
        row_names=row_names,
    )
