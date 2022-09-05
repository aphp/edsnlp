import re
from typing import Any, Callable, List, Union

from spacy.language import Language

from edsnlp.pipelines.ner.scores import Score
from edsnlp.pipelines.ner.scores.emergency.ccmu import patterns
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    regex=patterns.regex,
    value_extract=patterns.value_extract,
    score_normalization=patterns.score_normalization_str,
    attr="NORM",
    window=20,
    ignore_excluded=False,
    flags=0,
)


@deprecated_factory(
    "emergency.ccmu",
    "eds.emergency.ccmu",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
@Language.factory(
    "eds.emergency.ccmu",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    regex: List[str],
    value_extract: str,
    score_normalization: Union[str, Callable[[Union[str, None]], Any]],
    attr: str,
    window: int,
    ignore_excluded: bool,
    flags: Union[re.RegexFlag, int],
):
    return Score(
        nlp,
        score_name=name,
        regex=regex,
        value_extract=value_extract,
        score_normalization=score_normalization,
        attr=attr,
        window=window,
        ignore_excluded=ignore_excluded,
        flags=flags,
    )
