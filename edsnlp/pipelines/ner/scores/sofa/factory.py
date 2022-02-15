from typing import Any, Callable, List, Union

from spacy.language import Language

from edsnlp.pipelines.ner.scores.sofa import Sofa, patterns

sofa_default_config = dict(
    regex=patterns.regex,
    method_regex=patterns.method_regex,
    value_regex=patterns.value_regex,
    score_normalization=patterns.score_normalization_str,
)


@Language.factory("SOFA", default_config=sofa_default_config)
def create_component(
    nlp: Language,
    name: str,
    regex: List[str],
    method_regex: str,
    value_regex: str,
    score_normalization: Union[str, Callable[[Union[str, None]], Any]],
    attr: str = "NORM",
    window: int = 20,
    verbose: int = 0,
    ignore_excluded: bool = False,
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
