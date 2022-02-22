from typing import Optional

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .endlines import EndLines


@deprecated_factory("endlines", "eds.endlines")
@Language.factory("eds.endlines")
def create_component(
    nlp: Language,
    name: str,
    model_path: Optional[str],
):
    return EndLines(nlp, end_lines_model=model_path)
