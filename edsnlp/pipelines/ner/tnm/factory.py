from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .patterns import tnm_pattern
from .tnm import TNMMatcher

DEFAULT_CONFIG = dict(
    pattern=tnm_pattern,
    attr="TEXT",
    label="tnm",
    span_setter={"ents": True, "tnm": True},
)

create_component = TNMMatcher
create_component = deprecated_factory(
    "eds.TNM",
    "eds.tnm",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
create_component = Language.factory(
    "eds.tnm",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
