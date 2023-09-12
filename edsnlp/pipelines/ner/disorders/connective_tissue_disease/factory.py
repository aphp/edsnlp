from spacy import Language

from .connective_tissue_disease import ConnectiveTissueDiseaseMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="connective_tissue_disease",
    span_setter={"ents": True, "connective_tissue_disease": True},
)

create_component = Language.factory(
    "eds.connective_tissue_disease",
    assigns=["doc.ents", "doc.spans"],
)(ConnectiveTissueDiseaseMatcher)
