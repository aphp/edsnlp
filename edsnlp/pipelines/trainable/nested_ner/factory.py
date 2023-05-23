from spacy import Language
from thinc.api import Model
from thinc.config import Config

from .nested_ner import TrainableNer
from .nested_ner import make_nested_ner_scorer as create_scorer  # noqa: F401

nested_ner_default_config = """
[model]
    @architectures = "eds.stack_crf_ner_model.v1"
    mode = "joint"

    [model.tok2vec]
        @architectures = "spacy.Tok2Vec.v1"

    [model.tok2vec.embed]
        @architectures = "spacy.MultiHashEmbed.v1"
        width = 96
        rows = [5000, 2000, 1000, 1000]
        attrs = ["ORTH", "PREFIX", "SUFFIX", "SHAPE"]
        include_static_vectors = false

    [model.tok2vec.encode]
        @architectures = "spacy.MaxoutWindowEncoder.v1"
        width = ${model.tok2vec.embed.width}
        window_size = 1
        maxout_pieces = 3
        depth = 4

[scorer]
    @scorers = "eds.nested_ner_scorer.v1"
"""

NESTED_NER_DEFAULTS = Config().from_str(nested_ner_default_config)


@Language.factory(
    "nested_ner",
    default_config=NESTED_NER_DEFAULTS,
    requires=["doc.ents", "doc.spans"],
    assigns=["doc.ents", "doc.spans"],
    default_score_weights={
        "ents_f": 1.0,
        "ents_p": 0.0,
        "ents_r": 0.0,
    },
)
def create_component(
    nlp: Language,
    name: str,
    model: Model,
    ent_labels=None,
    spans_labels=None,
    scorer=None,
):
    """
    Initialize a general named entity recognizer (with or without nested or
    overlapping entities).

    Parameters
    ----------
    nlp: Language
        The current nlp object
    name: str
        Name of the component
    model: Model
        The model to extract the spans
    ent_labels: Iterable[str]
        list of labels to filter entities for in `doc.ents`
    spans_labels: Mapping[str, Iterable[str]]
        Mapping from span group names to list of labels to look for entities
        and assign the predicted entities
    scorer: Optional[Callable]
        Method to call to score predictions
    """
    return TrainableNer(
        vocab=nlp.vocab,
        model=model,
        name=name,
        ent_labels=ent_labels,
        spans_labels=spans_labels,
        scorer=scorer,
    )
