from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .history import HistoryQualifier

DEFAULT_CONFIG = dict(
    history=None,
    termination=None,
    use_sections=False,
    use_dates=False,
    attr="NORM",
    history_limit=14,
    closest_dates_only=True,
    exclude_birthdate=True,
    span_getter=None,
    on_ents_only=True,
    explain=False,
)

create_component = HistoryQualifier
for name in ["history", "antecedents", "eds.antecedents"]:
    create_component = deprecated_factory(
        name,
        "eds.history",
        assigns=["span._.history"],
    )(create_component)
create_component = Language.factory(
    "eds.history",
    assigns=["span._.history"],
)(create_component)
