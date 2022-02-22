from typing import Any, Callable, List, Union

from spacy.language import Language

from edsnlp.pipelines.ner.scores import Score
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    attr="NORM",
    window=7,
    verbose=0,
    ignore_excluded=False,
)


@deprecated_factory("score", "eds.score", default_config=DEFAULT_CONFIG)
@Language.factory("eds.score", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    score_name: str,
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
        score_name=score_name,
        regex=regex,
        after_extract=after_extract,
        score_normalization=score_normalization,
        attr=attr,
        window=window,
        verbose=verbose,
        ignore_excluded=ignore_excluded,
    )
