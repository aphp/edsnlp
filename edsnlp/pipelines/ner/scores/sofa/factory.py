from typing import Any, Callable, Dict, List, Union

import re
from spacy.language import Language

from edsnlp.pipelines.ner.scores.sofa import Sofa, patterns
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    regex=patterns.regex,
    after_extract=patterns.after_extract,
    score_normalization=patterns.score_normalization_str,
    attr="NORM",
    window=10,
    ignore_excluded=False,
    flags=re.S,
)


@deprecated_factory("SOFA", "eds.SOFA", default_config=DEFAULT_CONFIG)
@Language.factory("eds.SOFA", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    regex: List[str],
    after_extract: List[Dict[str, str]],
    score_normalization: Union[str, Callable[[Union[str, None]], Any]],
    attr: str,
    window: int,
    ignore_excluded: bool,
    flags: Union[re.RegexFlag, int],
):
    return Sofa(
        nlp,
        score_name=name,
        regex=regex,
        after_extract=after_extract,
        score_normalization=score_normalization,
        attr=attr,
        window=window,
        ignore_excluded=ignore_excluded,
        flags=flags,
    )
