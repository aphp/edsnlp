from typing import Optional

from spacy.language import Language

from edsnlp.pipelines import endlines
from edsnlp.pipelines.endlines import EndLines


@Language.factory("endlines")
def create_component(
    nlp: Language,
    name: str,
    model_path: Optional[str],
):
    return EndLines(nlp, end_lines_model=model_path)
