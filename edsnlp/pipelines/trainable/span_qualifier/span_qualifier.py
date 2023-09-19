from __future__ import annotations

import pickle
from collections import defaultdict
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
)

import torch
import torch.nn.functional as F
from spacy.tokens import Doc, Span
from typing_extensions import Literal, NotRequired, TypedDict

from edsnlp.core import PipelineProtocol
from edsnlp.core.torch_component import BatchInput, BatchOutput, TorchComponent
from edsnlp.pipelines.base import BaseComponent
from edsnlp.pipelines.trainable.embeddings.typing import (
    EmbeddingComponent,
)
from edsnlp.utils.bindings import (
    BINDING_GETTERS,
    BINDING_SETTERS,
    Binding,
    BindingCandidateGetterArg,
    get_candidates,
)
from edsnlp.utils.filter import align_spans
from edsnlp.utils.span_getters import get_spans

SpanQualifierBatchInput = TypedDict(
    "SpanQualifierBatchInput",
    {
        "embedding": BatchInput,
        "begins": torch.Tensor,
        "ends": torch.Tensor,
        "sequence_idx": torch.Tensor,
        "bindings_group_mask": torch.Tensor,
        "bindings": NotRequired[torch.Tensor],
    },
)
"""
embeds: torch.FloatTensor
    Token embeddings to predict the tags from
mask: torch.BoolTensor
    Mask of the sequences
spans: torch.LongTensor
    2d tensor of n_spans * (doc_idx, ner_label_idx, begin, end)
targets: NotRequired[List[torch.LongTensor]]
    list of 2d tensor of n_spans * n_combinations (1 hot)
"""


class TrainableSpanQualifier(
    TorchComponent[BatchOutput, SpanQualifierBatchInput], BaseComponent
):
    """
    The `eds.span_qualifier` component is a trainable qualifier, predictor of span
    attributes. In this context, the span qualification task consists in assigning values
     (boolean, strings or any complex object) to attributes/extensions of spans such as:

    - `span.label_`,
    - `span._.negation`,
    - `span._.date.mode`
    - etc.

    Architecture
    ------------
    The underlying `eds.span_multilabel_classifier.v1` model performs span classification by:

    1. Pooling the words embedding (`mean`, `max` or `sum`) into a single embedding per span
    2. Computing logits for each possible binding (i.e. qualifier-value assignment)
    3. Splitting these bindings into independent groups such as

        - `event_type=start` and `event_type=stop`
        - `negated=False` and `negated=True`

    4. Learning or predicting a combination amongst legal combination of these bindings.
    For instance in the second group, we can't have both `negated=True` and `negated=False` so the combinations are `[(1, 0), (0, 1)]`
    5. Assigning bindings on spans depending on the predicted results

    Step by step
    ------------
    ## Initialization
    During the initialization of the pipeline, the `span_qualifier` component will gather all spans
    that match `on_ents` and `on_span_groups` patterns (or `candidate_getter` function). It will then list
    all possible values for each `qualifier` of the `qualifiers` list and store every possible
    (qualifier, value) pair (i.e. binding).

    For instance, a custom qualifier `negation` with possible values `True` and `False` will result in the following bindings
    `[("_.negation", True), ("_.negation", False)]`, while a custom qualifier `event_type` with possible values `start`, `stop`, and `start-stop` will result in the following bindings `[("_.event_type", "start"), ("_.event_type", "stop"), ("_.event_type", "start-stop")]`.

    ## Training
    During training, the `span_qualifier` component will gather spans on the documents in a mini-batch
    and evaluate each binding on each span to build a supervision matrix.
    This matrix will be feed it to the underlying model (most likely a `eds.span_multilabel_classifier.v1`).
    The model will compute logits for each entry of the matrix and compute a cross-entropy loss for each group of bindings
    sharing the same qualifier. The loss will not be computed for entries that violate the `label_constraints` parameter (for instance, the `event_type` qualifier can only be assigned to spans with the `event` label).

    ## Prediction
    During prediction, the `span_qualifier` component will gather spans on a given document and evaluate each binding on each span using the underlying model. Using the same binding exclusion and label constraint mechanisms as during training, scores will be computed for each binding and the best legal combination of bindings will be selected. Finally, the selected bindings will be assigned to the spans.

    Parameters
    ----------
    nlp: PipelineProtocol
        Spacy vocabulary
    model: Model
        The model to extract the spans
    name: str
        Name of the component
    qualifiers: Optional[Sequence[str]]
        The qualifiers to predict or train on. If None, keys from the
        `label_constraints` will be used
    label_constraints: Optional[Dict[str, List[str]]]
        Constraints to select qualifiers for each span depending on their labels.
        Keys of the dict are the qualifiers and values are the labels for which
        the qualifier is allowed. If None, all qualifiers will be used for all spans
    candidate_getter: BindingCandidateGetterArg
        How to extract the candidate spans and the qualifiers
        to predict or train on.
    pooler_mode: Literal["max", "sum", "mean"]
        How embeddings are aggregated
    projection_mode: Literal["dot"]
        How embeddings converted into logits
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: str,
        *,
        embedding: EmbeddingComponent,
        candidate_getter: BindingCandidateGetterArg,
        pooler_mode: Literal["max", "sum", "mean"] = "mean",
        projection_mode: Literal["dot"] = "dot",
    ):
        self.qualifiers = None

        super().__init__(nlp, name)

        self.projection_mode = projection_mode
        self.pooler_mode = pooler_mode

        if projection_mode != "dot":
            raise Exception(
                "Only scalar product is supported for label classification."
            )

        self.embedding = embedding
        self.classifier = torch.nn.Linear(embedding.output_size, 0)
        self.label_constraints = []
        self.ner_labels_indices: Optional[Dict[str, int]] = None
        self.candidate_getter = candidate_getter

        self.bindings_indexer_per_group: List[slice] = []
        self.bindings_group_mask = None
        self.bindings: List[Binding] = []
        self.group_qualifiers: List[set] = []

    def to_disk(self, path, *, exclude=set()):
        if self.name in exclude:
            return
        exclude.add(self.name)
        self.embedding.to_disk(path, exclude=exclude)
        # This will receive the directory path + /my_component
        # We save the bindings as a pickle file since values can be arbitrary objects
        data_path = path / "bindings.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(
                {
                    "bindings": self.bindings,
                    "bindings_indexer_per_group": self.bindings_indexer_per_group,
                },
                f,
            )

    def from_disk(self, path, exclude=tuple()):
        super().from_disk(path, exclude=exclude)
        # This will receive the directory path + /my_component
        data_path = path / "bindings.pkl"
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.bindings = data["bindings"]
        self.bindings_indexer_per_group = data["bindings_indexer_per_group"]
        self.qualifiers = sorted(set(key for key, _ in self.bindings))
        self.set_extensions()
        return self

    def set_extensions(self):
        super().set_extensions()

        for qlf in self.qualifiers or ():
            if not Span.has_extension(qlf):
                Span.set_extension(qlf, default=None)

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]):
        super().post_init(gold_data, exclude=exclude)

        qualifier_values = defaultdict(set)
        for doc in gold_data:
            spans, spans_qualifiers = get_candidates(doc, self.candidate_getter)
            for span, span_qualifiers in zip(spans, spans_qualifiers):
                for qualifier in span_qualifiers:
                    value = BINDING_GETTERS[qualifier](span)
                    qualifier_values[qualifier].add(value)

        qualifier_values = {
            key: sorted(values, key=str) for key, values in qualifier_values.items()
        }

        # if self.qualifiers is not None and set(self.qualifiers) != set(
        #     qualifier_values.keys()
        # ):
        #     warnings.warn(
        #         f"Qualifiers {sorted(self.qualifiers)} do not match qualifiers found in "
        #         f"gold data {sorted(qualifier_values.keys())}"
        #     )
        self.qualifiers = sorted(set(qualifier_values))
        self.bindings = []
        self.bindings_indexer_per_group = []

        # `groups_combinations`:
        # - for each group (i.e. qualifier here)
        # - for each legal combination of bindings in this group
        # - bindings in this combination
        # ATM, create one group per qualifier (e.g. one group for _.event,
        # one for _.negation, ...), and so one binding per combination
        combinations_per_group = [
            [((key, value),) for value in sorted(values, key=str)]
            for key, values in qualifier_values.items()
        ]
        bindings_per_group = [
            list(dict.fromkeys(binding for comb in group for binding in comb))
            for group in combinations_per_group
        ]
        self.group_qualifiers = [
            list(dict.fromkeys(key for key, _ in group)) for group in bindings_per_group
        ]

        for grp_idx, (group_combinations, group_bindings) in enumerate(
            zip(combinations_per_group, bindings_per_group)
        ):
            self.bindings_indexer_per_group.append(
                slice(len(self.bindings), len(self.bindings) + len(group_bindings))
            )
            self.bindings.extend(group_bindings)
            # combinations buffer: bool tensor of shape
            #   num legal combinations
            # * num bindings in this group
            # TODO: use sparse tensor or None to mark identity combinations
            self.register_buffer(
                f"combinations_{grp_idx}",
                torch.as_tensor(
                    [
                        [binding in combination_bindings for binding in group_bindings]
                        for combination_bindings in group_combinations
                    ],
                    dtype=torch.bool,
                ).float(),
            )
        self.classifier = torch.nn.Linear(
            in_features=self.classifier.in_features,
            out_features=len(self.bindings),
        )

    @property
    def combinations_per_group(self):
        return (
            getattr(self, f"combinations_{i}")
            for i in range(len(self.bindings_indexer_per_group))
        )

    def _preprocess(self, doc, spans, spans_qualifiers):
        embedded_spans = list(get_spans(doc, self.embedding.span_getter))
        sequence_idx = []
        begins = []
        ends = []

        for i, (embedded_span, target_ents) in enumerate(
            zip(
                embedded_spans,
                align_spans(
                    source=spans,
                    target=embedded_spans,
                ),
            )
        ):
            start = embedded_span.start
            sequence_idx.extend([i] * len(spans))
            begins.extend([span.start - start for span in spans])
            ends.extend([span.end - start for span in spans])
        return {
            "begins": begins,
            "ends": ends,
            "sequence_idx": sequence_idx,
            "embedding": self.embedding.preprocess(doc),
            "bindings_group_mask": [
                [
                    not set(group_qualifiers).isdisjoint(set(span_qualifiers))
                    for group_qualifiers in self.group_qualifiers
                ]
                for span_qualifiers in spans_qualifiers
            ],
        }

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        spans, spans_qualifiers = get_candidates(doc, self.candidate_getter)
        # `spans`: [Span(doc1, 0, 3), Span(doc2, 0, 1), ...]
        # `spans_qualifiers`: [["_.attr1"], ["_.attr1", "_.attr3"], ...]

        return self._preprocess(doc, spans, spans_qualifiers)

    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        spans, spans_qualifiers = get_candidates(doc, self.candidate_getter)
        # `spans`: [Span(doc1, 0, 3), Span(doc2, 0, 1), ...]
        # `spans_qualifiers`: [["_.attr1"], ["_.attr1", "_.attr3"], ...]
        return {
            **self._preprocess(doc, spans, spans_qualifiers),
            "bindings": [
                [
                    binding[0] in span_qualifiers and BINDING_GETTERS[binding](span)
                    for binding in self.bindings
                ]
                for span, span_qualifiers in zip(spans, spans_qualifiers)
            ],
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> SpanQualifierBatchInput:
        collated: SpanQualifierBatchInput = {
            "embedding": self.embedding.collate(batch["embedding"]),
            # "sample_idx": torch.as_tensor(
            #     [i for i, x in enumerate(batch["begins"]) for b in x]
            # ),
            "begins": torch.as_tensor([b for x in batch["begins"] for b in x]),
            "ends": torch.as_tensor([e for x in batch["ends"] for e in x]),
            "sequence_idx": torch.as_tensor(
                [e for x in batch["sequence_idx"] for e in x]
            ),
            "bindings_group_mask": torch.as_tensor(
                [
                    span_bindings
                    for sample in batch["bindings_group_mask"]
                    for span_bindings in sample
                ]
            ),
        }
        if "bindings" in batch:
            collated["bindings"] = torch.as_tensor(
                [
                    span_bindings
                    for sample in batch["bindings"]
                    for span_bindings in sample
                ],
                dtype=torch.float,
            )
        return collated

    def pool_spans(
        self,
        embeddings: torch.Tensor,
        sample_idx: torch.Tensor,
        span_begins: torch.Tensor,
        span_ends: torch.Tensor,
    ) -> torch.FloatTensor:

        n_samples, n_words, dim = embeddings.shape
        device = embeddings.device

        flat_begins = n_words * sample_idx + span_begins
        flat_ends = n_words * sample_idx + span_ends
        flat_embeds = embeddings.view(-1, dim)
        flat_indices = torch.cat(
            [
                torch.arange(b, e, device=device)
                for b, e in zip(flat_begins.cpu().tolist(), flat_ends.cpu().tolist())
            ]
        ).to(device)
        offsets = (flat_ends - flat_begins).cumsum(0).roll(1)
        offsets[0] = 0
        return torch.nn.functional.embedding_bag(  # type: ignore
            input=flat_indices,
            weight=flat_embeds,
            offsets=offsets,
            mode=self.pooler_mode,
        )

    # noinspection SpellCheckingInspection
    def forward(self, batch: SpanQualifierBatchInput) -> BatchOutput:
        """
        Apply the span classifier module to the document embeddings and given spans to:
        - compute the loss
        - and/or predict the labels of spans
        If labels are predicted, they are assigned to the `additional_outputs`
        dictionary.

        Parameters
        ----------
        batch: SpanQualifierBatchInput
            The input batch

        Returns
        -------
        BatchOutput
        """
        device = next(self.parameters()).device
        if len(batch["begins"]) == 0:
            loss = pred = None
            if "bindings" in batch:
                loss = torch.zeros(0, device=device, requires_grad=True)
            else:
                pred = torch.zeros(0, self.n_labels, device=device, dtype=torch.int)
            return {
                "loss": loss,
                "labels": pred,
            }

        embeds = self.embedding.module_forward(batch["embedding"])
        span_embeds = self.pool_spans(
            embeds["embeddings"], batch["sequence_idx"], batch["begins"], batch["ends"]
        )

        n_spans, dim = span_embeds.shape
        n_bindings = len(self.bindings)
        device = span_embeds.device

        binding_scores = self.classifier(span_embeds)

        losses = []
        if "bindings" not in batch:
            pred = torch.zeros(n_spans, n_bindings, device=device)
        else:
            pred = None

        for group_idx, (bindings_indexer, combinations) in enumerate(
            zip(
                self.bindings_indexer_per_group,
                self.combinations_per_group,
            )
        ):
            group_samples_mask = batch["bindings_group_mask"][:, group_idx]
            combinations_scores = torch.einsum(
                "eb,cb->ec",
                binding_scores[group_samples_mask][:, bindings_indexer],
                combinations.float(),
            )

            if "bindings" in batch:
                group_samples_mask = batch["bindings_group_mask"][:, group_idx]
                # ([e]ntities * [b]indings) * ([c]ombinations * [b]indings)
                targets = torch.einsum(
                    "eb,cb->ec",
                    batch["bindings"][group_samples_mask][:, bindings_indexer],
                    combinations,
                )
                # [e]ntities * [c]comb --(argmax)--> [e]entities
                # Choose the combination of bindings in this group that fits the
                # most the gold bindings
                targets = targets.argmax(-1)
                losses.append(F.cross_entropy(combinations_scores, targets))
            elif "bindings" not in batch:
                pred[:, bindings_indexer][group_samples_mask] = combinations[
                    combinations_scores.argmax(-1)
                ]

        return {
            "loss": sum(losses) if losses else None,
            "labels": pred,
        }

    def get_candidate_spans(self, doc: Doc):
        return get_candidates(doc, self.candidate_getter)[0]

    def postprocess(self, docs: Sequence[Doc], batch: BatchOutput) -> Sequence[Doc]:
        spans = [span for doc in docs for span in self.get_candidate_spans(doc)]
        for span_idx, binding_idx in batch["labels"].nonzero(as_tuple=False):
            span = spans[span_idx]
            BINDING_SETTERS[self.bindings[binding_idx]](span)
        return docs
