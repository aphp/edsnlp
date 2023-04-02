from edsnlp.core import registry

from .adicap import AdicapMatcher
from .patterns import adicap_prefix, base_code

DEFAULT_CONFIG = dict(
    pattern=base_code,
    prefix=adicap_prefix,
    window=500,
    attr="TEXT",
    label="adicap",
    span_setter={"ents": True, "adicap": True},
)

create_component = registry.factory.register(
    "eds.adicap",
    assigns=["doc.ents", "doc.spans"],
)(AdicapMatcher)
