from edsnlp.core import registry

from .relations import RelationsMatcher

DEFAULT_CONFIG = dict(
    scheme=None,
    use_sentences=False,
    clean_rel=False,
    proximity_method = "right",
    max_dist=45,
)

create_component = registry.factory.register(
    "eds.relations",
    assigns=["doc.spans"],
    deprecated=["relations"],
)(RelationsMatcher)