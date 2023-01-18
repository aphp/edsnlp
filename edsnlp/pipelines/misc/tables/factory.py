from typing import Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.misc.tables import Tables
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    tables_pattern=None,
    sep_pattern=None,
    attr="TEXT",
    ignore_excluded=False,
)


@deprecated_factory("tables", "eds.tables", default_config=DEFAULT_CONFIG)
@Language.factory("eds.tables", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    tables_pattern: Optional[Dict[str, Union[List[str], str]]],
    sep_pattern: Optional[str],
    attr: str,
    ignore_excluded: bool,
):
    return Tables(
        nlp,
        tables_pattern=tables_pattern,
        sep_pattern=sep_pattern,
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
