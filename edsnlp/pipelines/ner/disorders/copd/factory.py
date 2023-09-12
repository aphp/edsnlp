from spacy import Language

from edsnlp.utils.deprecation import deprecated_factory

from .copd import COPDMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="copd",
    span_setter={"ents": True, "copd": True},
)

create_component = deprecated_factory(
    "eds.COPD",
    "eds.copd",
    assigns=["doc.ents", "doc.spans"],
)(COPDMatcher)
create_component = Language.factory(
    "eds.copd",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
