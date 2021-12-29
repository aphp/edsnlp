from typing import Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.reason import Reason, terms

reason_default_config = dict(
    regex=terms.reasons,
    attr="TEXT",
    terms=None,
    use_sections=False,
    ignore_excluded=False,
)


@Language.factory("reason", default_config=reason_default_config)
def create_component(
    nlp: Language,
    name: str,
    regex: Dict[str, Union[List[str], str]],
    attr: str,
    use_sections: bool,
    terms: Optional[Dict[str, Union[List[str], str]]],
    ignore_excluded: bool,
):
    return Reason(
        nlp,
        terms=terms,
        attr=attr,
        regex=regex,
        use_sections=use_sections,
        ignore_excluded=ignore_excluded,
    )
