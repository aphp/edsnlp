from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .patterns import sections
from .sections import SectionsMatcher

DEFAULT_CONFIG = dict(
    sections=sections,
    add_patterns=True,
    attr="NORM",
    ignore_excluded=True,
)

create_component = deprecated_factory(
    "sections",
    "eds.sections",
    assigns=["doc.spans", "doc.ents"],
)(SectionsMatcher)
create_component = Language.factory(
    "eds.sections",
    assigns=["doc.spans", "doc.ents"],
)(create_component)
