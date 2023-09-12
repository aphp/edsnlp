from spacy import Language

from edsnlp.utils.deprecation import deprecated_factory

from .aids import AIDSMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="aids",
    span_setter={"ents": True, "aids": True},
)

create_component = deprecated_factory(
    "eds.AIDS",
    "eds.aids",
    assigns=["doc.ents", "doc.spans"],
)(AIDSMatcher)
create_component = Language.factory(
    "eds.aids",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
