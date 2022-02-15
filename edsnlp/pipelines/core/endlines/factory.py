from typing import Optional

from spacy.language import Language

from .endlines import EndLines


@Language.factory("endlines")
def create_component(
    nlp: Language,
    name: str,
    model_path: Optional[str],
):
    return EndLines(nlp, end_lines_model=model_path)
