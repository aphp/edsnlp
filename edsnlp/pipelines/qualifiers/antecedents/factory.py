from typing import List, Optional

from spacy.language import Language

from edsnlp.pipelines.qualifiers.antecedents import Antecedents, patterns
from edsnlp.pipelines.terminations import termination

antecedents_default_config = dict(
    attr="NORM",
    antecedents=patterns.antecedents,
    termination=termination,
    use_sections=False,
    explain=False,
    on_ents_only=True,
)


@Language.factory("antecedents", default_config=antecedents_default_config)
def create_component(
    nlp: Language,
    name: str,
    antecedents: Optional[List[str]],
    termination: Optional[List[str]],
    use_sections: bool,
    attr: str,
    explain: str,
    on_ents_only: bool,
):
    return Antecedents(
        nlp,
        attr=attr,
        antecedents=antecedents,
        termination=termination,
        use_sections=use_sections,
        explain=explain,
        on_ents_only=on_ents_only,
    )
