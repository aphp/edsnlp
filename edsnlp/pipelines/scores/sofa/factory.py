from typing import Any, Callable, List, Union

from spacy.language import Language

from edsnlp.pipelines.scores.sofa import Sofa, terms

sofa_default_config = dict(
    regex=terms.regex,
    method_regex=terms.method_regex,
    value_regex=terms.value_regex,
    score_normalization=terms.score_normalization_str,
)


@Language.factory("SOFA", default_config=sofa_default_config)
def create_component(
    nlp: Language,
    name: str,
    regex: List[str],
    method_regex: str,
    value_regex: str,
    score_normalization: Union[str, Callable[[Union[str, None]], Any]],
    attr: str = "CUSTOM_NORM",
    window: int = 20,
    verbose: int = 0,
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
    )
