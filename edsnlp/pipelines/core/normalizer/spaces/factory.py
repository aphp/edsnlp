from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .spaces import SpacesTagger

DEFAULT_CONFIG = dict(newline=True)

create_component = SpacesTagger
create_component = deprecated_factory(
    "spaces",
    "eds.spaces",
    assigns=["token.tag"],
)(create_component)
create_component = Language.factory(
    "eds.spaces",
    assigns=["token.tag"],
)(create_component)
