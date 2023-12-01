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
    use_dates=False,
    history_limit=14,
    exclude_birthdate=True,
    closest_dates_only=True,
    explain=False,
    on_ents_only=True,
)


@deprecated_factory(
    "antecedents",
    "eds.history",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.history"],
)
@deprecated_factory(
    "eds.antecedents",
    "eds.history",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.history"],
)
@deprecated_factory(
    "history",
    "eds.history",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.history"],
)
@Language.factory(
    "eds.history",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.history"],
)
def create_component(
    nlp: Language,
    name: str,
    history: Optional[List[str]],
    termination: Optional[List[str]],
    use_sections: bool,
    use_dates: bool,
    history_limit: int,
    exclude_birthdate: bool,
    closest_dates_only: bool,
    attr: str,
    explain: bool,
    on_ents_only: bool,
):
    return History(
        nlp,
        attr=attr,
        history=history,
        termination=termination,
        use_sections=use_sections,
        use_dates=use_dates,
        history_limit=history_limit,
        exclude_birthdate=exclude_birthdate,
        closest_dates_only=closest_dates_only,
        explain=explain,
        on_ents_only=on_ents_only,
    )
