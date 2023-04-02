from edsnlp.core import registry
from edsnlp.utils.deprecation import deprecated_factory

from .endlines import EndLinesMatcher

DEFAULT_CONFIG = dict(
    model_path=None,
)

create_component = EndLinesMatcher
create_component = deprecated_factory(
    "endlines",
    "eds.endlines",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
create_component = registry.factory.register(
    "eds.endlines",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
