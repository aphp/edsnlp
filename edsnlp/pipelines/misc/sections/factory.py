from edsnlp.core import registry

from .patterns import sections
from .sections import SectionsMatcher

DEFAULT_CONFIG = dict(
    sections=sections,
    add_patterns=True,
    attr="NORM",
    ignore_excluded=True,
)

create_component = registry.factory.register(
    "eds.sections",
    assigns=["doc.spans", "doc.ents"],
    deprecated=["sections"],
)(SectionsMatcher)
