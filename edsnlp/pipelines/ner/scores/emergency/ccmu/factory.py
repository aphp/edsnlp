from typing import Any, Callable, List, Union

from spacy.language import Language

from edsnlp.pipelines.ner.scores import Score
from edsnlp.pipelines.ner.scores.emergency.ccmu import patterns
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    regex=patterns.regex,
    after_extract=patterns.after_extract,
    score_normalization=patterns.score_normalization_str,
    attr="NORM",
    window=20,
    verbose=0,
    ignore_excluded=False,
)


@deprecated_factory(
    "emergency.ccmu", "eds.emergency.ccmu", default_config=DEFAULT_CONFIG
)
@Language.factory("eds.emergency.ccmu", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    regex: List[str],
    after_extract: str,
    score_normalization: Union[str, Callable[[Union[str, None]], Any]],
    attr: str,
    window: int,
    verbose: int,
    ignore_excluded: bool,
):
    return Score(
        nlp,
        score_name=name,
        regex=regex,
        after_extract=after_extract,
        score_normalization=score_normalization,
        attr=attr,
        window=window,
        verbose=verbose,
        ignore_excluded=ignore_excluded,
    )
