from edsnlp.core import registry

from .spaces import SpacesTagger

DEFAULT_CONFIG = dict(newline=True)

create_component = registry.factory.register(
    "eds.spaces",
    assigns=["token.tag"],
    deprecated=["spaces"],
)(SpacesTagger)
