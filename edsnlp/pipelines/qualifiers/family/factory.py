from typing import List, Optional

from spacy.language import Language

from edsnlp.pipelines.qualifiers.family import FamilyContext
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    family=None,
    termination=None,
    attr="NORM",
    use_sections=False,
    explain=False,
    on_ents_only=True,
)


@deprecated_factory(
    "family",
    "eds.family",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.family"],
)
@Language.factory(
    "eds.family",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.family"],
)
def create_component(
    nlp: Language,
    name: str,
    family: Optional[List[str]],
    termination: Optional[List[str]],
    attr: str,
    explain: bool,
    on_ents_only: bool,
    use_sections: bool,
):
    return FamilyContext(
        nlp,
        family=family,
        termination=termination,
        attr=attr,
        explain=explain,
        on_ents_only=on_ents_only,
        use_sections=use_sections,
    )
