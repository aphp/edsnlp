from edsnlp.core import registry

from .patterns_new import tnm_pattern_new
from .tnm import TNMMatcher

DEFAULT_CONFIG = dict(
    pattern=tnm_pattern_new,
    attr="TEXT",
    label="tnm",
    span_setter={"ents": True, "tnm": True},
)

create_component = registry.factory.register(
    "eds.tnm",
    assigns=["doc.ents", "doc.spans"],
    deprecated=["eds.TNM"],
)(TNMMatcher)
