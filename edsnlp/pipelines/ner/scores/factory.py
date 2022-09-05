import re
from typing import Any, Callable, List, Union

from spacy.language import Language

from edsnlp.pipelines.ner.scores import Score
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    attr="NORM",
    window=7,
    ignore_excluded=False,
    flags=0,
)


@deprecated_factory(
    "score",
    "eds.score",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
@Language.factory(
    "eds.score",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    score_name: str,
    regex: List[str],
    value_extract: str,
    score_normalization: Union[str, Callable[[Union[str, None]], Any]],
    attr: str,
    window: int,
    flags: Union[re.RegexFlag, int],
    ignore_excluded: bool,
):
    return Score(
        nlp,
        score_name=score_name,
        regex=regex,
        value_extract=value_extract,
        score_normalization=score_normalization,
        attr=attr,
        flags=flags,
        window=window,
        ignore_excluded=ignore_excluded,
    )
