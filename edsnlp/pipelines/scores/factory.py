from typing import Any, Callable, List, Union

from spacy.language import Language

from edsnlp.pipelines.scores import Score


@Language.factory("score", default_config=dict())
def create_component(
    nlp: Language,
    name: str,
    score_name: str,
    regex: List[str],
    after_extract: str,
    score_normalization: Union[str, Callable[[Union[str, None]], Any]],
    attr: str = "NORM",
    window: int = 7,
    verbose: int = 0,
    ignore_excluded: bool = False,
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
