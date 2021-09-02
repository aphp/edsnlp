from typing import Dict, Any, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.antecedents import Antecedents, terms
from edsnlp.pipelines.terminations import termination


antecedents_default_config = dict(
    antecedents=terms.antecedents,
    termination=termination,
)


@Language.factory("antecedents", default_config=antecedents_default_config)
def create_antecedents_component(
    nlp: Language,
    name: str,
    antecedents: List[str],
    termination: List[str],
    use_sections: bool = True,
    fuzzy: bool = False,
    filter_matches: bool = False,
    annotation_scheme: str = "all",
    attr: str = "LOWER",
    on_ents_only: bool = True,
    regex: Optional[Dict[str, Union[List[str], str]]] = None,
    fuzzy_kwargs: Optional[Dict[str, Any]] = None,
):
    return Antecedents(
        nlp,
        antecedents=antecedents,
        termination=termination,
        use_sections=use_sections,
        fuzzy=fuzzy,
        filter_matches=filter_matches,
        annotation_scheme=annotation_scheme,
        attr=attr,
        on_ents_only=on_ents_only,
        regex=regex,
        fuzzy_kwargs=fuzzy_kwargs,
    )
