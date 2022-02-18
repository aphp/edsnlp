from typing import Any, Callable, List, Union

from spacy.language import Language

from edsnlp.pipelines.ner.scores import Score
from edsnlp.pipelines.ner.scores.emergency.priority import patterns

priority_default_config = dict(
    regex=patterns.regex,
    after_extract=patterns.after_extract,
    score_normalization=patterns.score_normalization_str,
)


@Language.factory("emergency.priority", default_config=priority_default_config)
def create_component(
    nlp: Language,
    name: str,
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
        score_name=name,
        regex=regex,
        after_extract=after_extract,
        score_normalization=score_normalization,
        attr=attr,
        window=window,
        verbose=verbose,
        ignore_excluded=ignore_excluded,
    )
