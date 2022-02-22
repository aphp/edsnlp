from typing import Any, Callable, List, Union

from spacy.language import Language

from edsnlp.pipelines.ner.scores.sofa import Sofa, patterns
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    regex=patterns.regex,
    method_regex=patterns.method_regex,
    value_regex=patterns.value_regex,
    score_normalization=patterns.score_normalization_str,
    attr="NORM",
    window=20,
    verbose=0,
    ignore_excluded=False,
)


@deprecated_factory("SOFA", "eds.SOFA", default_config=DEFAULT_CONFIG)
@Language.factory("eds.SOFA", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    regex: List[str],
    method_regex: str,
    value_regex: str,
    score_normalization: Union[str, Callable[[Union[str, None]], Any]],
    attr: str,
    window: int,
    verbose: int,
    ignore_excluded: bool,
):
    return Sofa(
        nlp,
        score_name=name,
        regex=regex,
        method_regex=method_regex,
        value_regex=value_regex,
        score_normalization=score_normalization,
        attr=attr,
        window=window,
        verbose=verbose,
        ignore_excluded=ignore_excluded,
    )
