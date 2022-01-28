from typing import List, Optional

from spacy.language import Language

from edsnlp.qualifiers.family import FamilyContext

family_default_config = dict(
    family=None,
    termination=None,
    attr="NORM",
    use_sections=False,
    explain=False,
    on_ents_only=True,
)


@Language.factory("family", default_config=family_default_config)
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
