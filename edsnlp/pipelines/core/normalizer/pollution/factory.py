from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .patterns import default_enabled
from .pollution import PollutionTagger

DEFAULT_CONFIG = dict(
    pollution=default_enabled,
)

create_component = PollutionTagger
create_component = deprecated_factory(
    "pollution",
    "eds.pollution",
    assigns=["doc.spans"],
)(create_component)
create_component = Language.factory(
    "eds.pollution",
    assigns=["doc.spans"],
)(create_component)
