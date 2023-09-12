from spacy.language import Language

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
create_component = Language.factory(
    "eds.endlines",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
