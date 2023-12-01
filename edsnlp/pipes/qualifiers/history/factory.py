from edsnlp.core import registry

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

create_component = registry.factory.register(
    "eds.history",
    assigns=["span._.history"],
    deprecated=[
        "history",
        "antecedents",
        "eds.antecedents",
    ],
)(HistoryQualifier)
