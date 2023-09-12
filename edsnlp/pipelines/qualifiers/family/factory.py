from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

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

create_component = deprecated_factory(
    "family",
    "eds.family",
    assigns=["span._.family"],
)(FamilyContextQualifier)
create_component = Language.factory(
    "eds.family",
    assigns=["span._.family"],
)(create_component)
