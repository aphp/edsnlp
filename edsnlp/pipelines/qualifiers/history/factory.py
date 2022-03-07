from typing import List, Optional

from spacy.language import Language

from edsnlp.pipelines.qualifiers.history import History, patterns
from edsnlp.pipelines.terminations import termination
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    attr="NORM",
    history=patterns.history,
    termination=termination,
    use_sections=False,
    explain=False,
    on_ents_only=True,
)


@deprecated_factory("antecedents", "eds.history", default_config=DEFAULT_CONFIG)
@deprecated_factory("eds.antecedents", "eds.history", default_config=DEFAULT_CONFIG)
@deprecated_factory("history", "eds.history", default_config=DEFAULT_CONFIG)
@Language.factory("eds.history", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    history: Optional[List[str]],
    termination: Optional[List[str]],
    use_sections: bool,
    attr: str,
    explain: str,
    on_ents_only: bool,
):
    return History(
        nlp,
        attr=attr,
        history=history,
        termination=termination,
        use_sections=use_sections,
        explain=explain,
        on_ents_only=on_ents_only,
    )
