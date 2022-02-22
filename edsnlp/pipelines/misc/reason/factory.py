from typing import Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.misc.reason import Reason, patterns
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    regex=patterns.reasons,
    attr="TEXT",
    terms=None,
    use_sections=False,
    ignore_excluded=False,
)


@deprecated_factory("reason", "eds.reason", default_config=DEFAULT_CONFIG)
@Language.factory("eds.reason", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    regex: Dict[str, Union[List[str], str]],
    attr: str,
    use_sections: bool,
    terms: Optional[Dict[str, Union[List[str], str]]],
    ignore_excluded: bool,
):
    return Reason(
        nlp,
        terms=terms,
        attr=attr,
        regex=regex,
        use_sections=use_sections,
        ignore_excluded=ignore_excluded,
    )
