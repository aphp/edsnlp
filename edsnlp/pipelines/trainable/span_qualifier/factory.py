from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from spacy import Language
from spacy.tokens import Doc
from thinc.api import Model
from thinc.config import Config

from .span_qualifier import TrainableSpanQualifier
from .span_qualifier import make_span_qualifier_scorer as create_scorer  # noqa: F401
from .utils import SpanGroups, Spans
from .utils import make_candidate_getter as create_candidate_getter

span_qualifier_default_config = """
[model]
    @architectures = "eds.span_multi_classifier.v1"
    projection_mode = "dot"
    pooler_mode = "max"
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
"""

SPAN_QUALIFIER_DEFAULTS = Config().from_str(span_qualifier_default_config)


@Language.factory(
    "eds.span_qualifier",
    default_config=SPAN_QUALIFIER_DEFAULTS,
    requires=["doc.ents", "doc.spans"],
    assigns=["doc.ents", "doc.spans"],
    default_score_weights={
        "qual_f": 1.0,
    },
)
def create_component(
    nlp,
    model: Model,
    on_ents: Optional[Union[bool, Sequence[str]]] = None,
    on_span_groups: Union[
        bool, Sequence[str], Mapping[str, Union[bool, Sequence[str]]]
    ] = False,
    qualifiers: Optional[Sequence[str]] = None,
    label_constraints: Optional[Dict[str, List[str]]] = None,
    candidate_getter: Optional[
        Callable[[Doc], Tuple[Spans, Optional[Spans], SpanGroups, List[List[str]]]]
    ] = None,
    name: str = "span_qualifier",
    scorer: Optional[Callable] = None,
) -> TrainableSpanQualifier:
    """
    Create a generic span classification component

    Parameters
    ----------
    nlp: Language
        Spacy vocabulary
    model: Model
        The model to extract the spans
    name: str
        Name of the component
    on_ents: Union[bool, Sequence[str]]
        Whether to look into `doc.ents` for spans to classify. If a list of strings
        is provided, only the span of the given labels will be considered. If None
        and `on_span_groups` is False, labels mentioned in `label_constraints`
        will be used, and all ents will be used if `label_constraints` is None.
    on_span_groups: Union[bool, Sequence[str], Mapping[str, Sequence[str]]]
        Whether to look into `doc.spans` for spans to classify:

        - If True, all span groups will be considered
        - If False, no span group will be considered
        - If a list of str is provided, only these span groups will be kept
        - If a mapping is provided, the keys are the span group names and the values
          are either a list of allowed labels in the group or True to keep them all
    qualifiers: Optional[Sequence[str]]
        The qualifiers to predict or train on. If None, keys from the
        `label_constraints` will be used
    label_constraints: Optional[Dict[str, List[str]]]
        Constraints to select qualifiers for each span depending on their labels.
        Keys of the dict are the qualifiers and values are the labels for which
        the qualifier is allowed. If None, all qualifiers will be used for all spans
    candidate_getter: Optional[Callable[[Doc], Tuple[Spans, Optional[Spans], SpanGroups, List[List[str]]]]]
        Optional method to call to extract the candidate spans and the qualifiers
        to predict or train on. If None, a candidate getter will be created from
        the other parameters: `on_ents`, `on_span_groups`, `qualifiers` and
        `label_constraints`.
    scorer: Optional[Callable]
        Optional method to call to score predictions
    """  # noqa: E501
    do_make_candidate_getter = (
        on_ents or on_span_groups or qualifiers or label_constraints
    )
    if (candidate_getter is not None) == do_make_candidate_getter:
        raise ValueError(
            "You must either provide a candidate getter or the parameters to "
            "make one, but not both."
        )
    if do_make_candidate_getter:
        candidate_getter = create_candidate_getter(
            on_ents=on_ents,
            on_span_groups=on_span_groups,
            qualifiers=qualifiers,
            label_constraints=label_constraints,
        )

    return TrainableSpanQualifier(
        vocab=nlp.vocab,
        model=model,
        candidate_getter=candidate_getter,
        name=name,
        scorer=scorer,
    )
