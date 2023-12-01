from edsnlp.core import registry

from .family import FamilyContextQualifier

DEFAULT_CONFIG = dict(
    attr="NORM",
    family=None,
    termination=None,
    use_sections=True,
    span_getter=None,
    on_ents_only=True,
    explain=False,
)

create_component = registry.factory.register(
    "eds.family",
    assigns=["span._.family"],
    deprecated=["family"],
)(FamilyContextQualifier)
