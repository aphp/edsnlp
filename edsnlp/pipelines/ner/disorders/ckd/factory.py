from spacy import Language

from edsnlp.utils.deprecation import deprecated_factory

from .ckd import CKDMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="ckd",
    span_setter={"ents": True, "ckd": True},
)

create_component = deprecated_factory(
    "eds.CKD",
    "eds.ckd",
    assigns=["doc.ents", "doc.spans"],
)(CKDMatcher)
create_component = Language.factory(
    "eds.ckd",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
