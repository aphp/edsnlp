from typing import Any, Dict, Union

from spacy.language import Language

from edsnlp.pipelines.core.terminology import TerminologyMatcher

from . import patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
    algorithm="exact",
    algorithm_config={},
)


@Language.factory(
    "eds.cim10", default_config=DEFAULT_CONFIG, assigns=["doc.ents", "doc.spans"]
)
def create_component(
    nlp: Language,
    name: str,
    attr: Union[str, Dict[str, str]],
    ignore_excluded: bool,
    algorithm: str,
    algorithm_config: Dict[str, Any],
):

    return TerminologyMatcher(
        nlp,
        label="cim10",
        regex=None,
        terms=patterns.get_patterns(),
        attr=attr,
        ignore_excluded=ignore_excluded,
        algorithm=algorithm,
        algorithm_config=algorithm_config,
    )
