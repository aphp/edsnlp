from edsnlp.core import registry
from edsnlp.utils.deprecation import deprecated_factory

from .spaces import SpacesTagger

DEFAULT_CONFIG = dict(newline=True)

create_component = SpacesTagger
create_component = deprecated_factory(
    "spaces",
    "eds.spaces",
    assigns=["token.tag"],
)(create_component)
create_component = registry.factory.register(
    "eds.spaces",
    assigns=["token.tag"],
)(create_component)
