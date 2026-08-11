from __future__ import annotations

import math
from collections import defaultdict
from itertools import accumulate
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from spacy.tokens import Doc, Span
from typing_extensions import NotRequired, TypedDict

from edsnlp.core.pipeline import Pipeline
from edsnlp.core.torch_component import BatchInput, TorchComponent
from edsnlp.pipes.base import BaseNERComponent
from edsnlp.pipes.trainable.embeddings.typing import WordEmbeddingComponent
from edsnlp.pipes.trainable.layers.crf import MultiLabelBIOULDecoder
from edsnlp.utils.span_getters import (
    SpanGetterArg,
    SpanSetterArg,
    get_spans,
    validate_span_getter,
)

EXTRACTION_MARKER = "[EXTRACT]"
CLASSIFICATION_MARKER = "[CLASSIFY]"
ROLE_MARKERS = [
    EXTRACTION_MARKER,
    CLASSIFICATION_MARKER,
]


def span_label(span: Span) -> str:
    """Expose the extraction id stored as the output span label"""

    return span.label_


MultiQueryQABatchInput = TypedDict(
    "MultiQueryQABatchInput",
    {
        "embedding": BatchInput,
        "lengths": torch.Tensor,
        "word_offsets": torch.Tensor,
        "extraction_context_indices": torch.Tensor,
        "extraction_prompt_indices": torch.Tensor,
        "extraction_ids": List[str],
        "classification_group_contexts": torch.Tensor,
        "classification_group_cardinalities": torch.Tensor,
        "classification_group_option_counts": torch.Tensor,
        "classification_option_prompt_indices": torch.Tensor,
        "classification_facets": List[str],
        "classification_value_ids": List[List[str]],
        "gold_tags": NotRequired[torch.Tensor],
        "gold_spans": NotRequired[torch.Tensor],
        "gold_extraction_pairs": NotRequired[torch.Tensor],
        "gold_group_candidate_indices": NotRequired[torch.Tensor],
        "gold_group_indices": NotRequired[torch.Tensor],
        "gold_single_targets": NotRequired[torch.Tensor],
        "gold_option_group_pair_indices": NotRequired[torch.Tensor],
        "gold_option_offsets": NotRequired[torch.Tensor],
        "gold_multi_targets": NotRequired[torch.Tensor],
        "stats": Dict[str, int],
    },
)

MultiQueryQABatchOutput = TypedDict(
    "MultiQueryQABatchOutput",
    {
        "loss": Optional[torch.Tensor],
        "candidate_loss": Optional[torch.Tensor],
        "extraction_loss": Optional[torch.Tensor],
        "classification_loss": Optional[torch.Tensor],
        "tags": torch.Tensor,
        "candidate_spans": torch.Tensor,
        "extraction_ids": List[str],
        "extraction_pair_candidates": torch.Tensor,
        "extraction_pair_queries": torch.Tensor,
        "extraction_selected": Optional[torch.Tensor],
        "classification_pair_candidates": torch.Tensor,
        "classification_pair_groups": torch.Tensor,
        "classification_score_pairs": torch.Tensor,
        "classification_score_options": torch.Tensor,
        "classification_selected": Optional[torch.Tensor],
        "classification_group_option_starts": torch.Tensor,
        "classification_facets": List[str],
        "classification_value_ids": List[List[str]],
    },
)


def make_union_tags(
    spans: torch.Tensor,
    num_contexts: int,
    max_words: int,
) -> tuple[torch.Tensor, int]:
    """Aggregate overlapping gold spans with the nlstruct BIOUL max rule"""

    tags = torch.zeros((num_contexts, max_words), dtype=torch.long)
    if not len(spans) or not max_words:
        return tags, 0

    contexts, begins, ends = spans.unbind(1)
    positions = torch.arange(max_words)
    flat_positions = contexts[:, None] * max_words + positions
    inside = (positions >= begins[:, None]) & (positions < ends[:, None])
    flat_tags = tags.flatten()
    flat_tags.scatter_reduce_(
        0,
        flat_positions[inside],
        torch.ones_like(flat_positions[inside]),
        reduce="amax",
    )

    singleton = ends - begins == 1
    boundary_indices = torch.cat(
        (
            contexts * max_words + begins,
            contexts * max_words + ends - 1,
        )
    )
    boundary_tags = torch.cat(
        (
            torch.where(singleton, 4, 2),
            torch.where(singleton, 4, 3),
        )
    )
    flat_tags.scatter_reduce_(0, boundary_indices, boundary_tags, reduce="amax")

    # Count words receiving incompatible roles while retaining every scorer target
    roles = torch.zeros((num_contexts * max_words, 5), dtype=torch.bool)
    strict_inside = (positions > begins[:, None]) & (positions < ends[:, None] - 1)
    roles[flat_positions[strict_inside], 1] = True
    roles[contexts * max_words + begins, torch.where(singleton, 4, 2)] = True
    roles[contexts * max_words + ends - 1, torch.where(singleton, 4, 3)] = True
    conflicts = int((roles[:, 2] & roles[:, 3] & ~roles[:, 4]).sum())
    return tags, conflicts


def pair_by_context(
    candidate_contexts: torch.Tensor,
    item_contexts: torch.Tensor,
    num_contexts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build packed candidate item Cartesian products within each context"""

    counts = torch.bincount(item_contexts, minlength=num_contexts)
    item_starts = counts.cumsum(0) - counts
    repeats = counts[candidate_contexts]
    pair_candidates = torch.arange(
        len(candidate_contexts), device=candidate_contexts.device
    ).repeat_interleave(repeats)
    pair_starts = repeats.cumsum(0) - repeats
    offsets = torch.arange(
        len(pair_candidates), device=candidate_contexts.device
    ) - pair_starts.repeat_interleave(repeats)
    pair_items = item_starts[candidate_contexts[pair_candidates]] + offsets
    return pair_candidates, pair_items, pair_starts, item_starts


class MultiQueryQAHead(torch.nn.Module):
    """Shared candidate detector and span prompt scorer"""

    def __init__(self, input_size: int, projection_size: int = 128):
        super().__init__()
        self.projection_size = projection_size
        self.candidate_linear = torch.nn.Linear(input_size, 5)
        self.span_projection = torch.nn.Linear(input_size * 3, projection_size)
        self.query_projection = torch.nn.Linear(input_size, projection_size)
        self.score_bias = torch.nn.Parameter(torch.zeros(()))
        self.crf = MultiLabelBIOULDecoder(
            1,
            learnable_transitions=False,
            allow_overlap=True,
        )

    def encode_spans(
        self,
        words: torch.Tensor,
        spans: torch.Tensor,
        word_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Project packed first last mean span vectors for all scorer candidates"""

        contexts, begins, ends = spans.unbind(1)
        begins = word_offsets[contexts] + begins
        ends = word_offsets[contexts] + ends
        prefix = F.pad(words.cumsum(0), (0, 0, 1, 0))
        means = (prefix[ends] - prefix[begins]) / (ends - begins).unsqueeze(1)
        return self.span_projection(
            torch.cat((words[begins], words[ends - 1], means), dim=1)
        )

    def encode_queries(
        self,
        prompt_embeddings: torch.Tensor,
        _prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project the contextualized role marker of each prompt item"""

        return self.query_projection(prompt_embeddings[:, 0])

    def score_pairs(
        self,
        span_features: torch.Tensor,
        query_features: torch.Tensor,
        span_indices: torch.Tensor,
        prompt_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Score packed span prompt pairs with one bilinear dot scorer"""

        return (span_features[span_indices] * query_features[prompt_indices]).sum(
            -1
        ) / math.sqrt(self.projection_size) + self.score_bias


class TrainableMultiQueryQA(
    TorchComponent[MultiQueryQABatchOutput, MultiQueryQABatchInput],
    BaseNERComponent,
):
    """Extract and classify spans for many dynamic queries in one report pass"""

    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: str = "multi_query_qa",
        *,
        embedding: WordEmbeddingComponent,
        target_span_getter: SpanGetterArg = "multi_query_qa_gold",
        span_setter: SpanSetterArg = "multi_query_qa",
        query_attribute: str = "queries",
        extraction_ids_attribute: str = "extraction_ids",
        extraction_id_attribute: str = "extraction_id",
        facet_values_attribute: str = "facet_values",
        facet_known_values_attribute: str = "facet_known_values",
        projection_size: int = 128,
        extraction_threshold: float = 0.5,
        multi_label_threshold: float = 0.5,
    ):
        self.target_span_getter = validate_span_getter(target_span_getter)
        self.query_attribute = query_attribute
        self.extraction_ids_attribute = extraction_ids_attribute
        self.extraction_id_attribute = extraction_id_attribute
        self.facet_values_attribute = facet_values_attribute
        self.facet_known_values_attribute = facet_known_values_attribute
        self.projection_size = projection_size
        self.extraction_threshold = extraction_threshold
        self.multi_label_threshold = multi_label_threshold
        super().__init__(nlp=nlp, name=name, span_setter=span_setter)
        self.embedding = embedding
        if hasattr(embedding, "add_special_tokens"):
            embedding.add_special_tokens(ROLE_MARKERS)
        self.head = MultiQueryQAHead(embedding.output_size, projection_size)

    def set_extensions(self):
        """Register the dynamic query and gold prediction span attributes"""

        super().set_extensions()
        if not Doc.has_extension(self.query_attribute):
            Doc.set_extension(self.query_attribute, default=None)
        if not Span.has_extension(self.extraction_id_attribute):
            Span.set_extension(self.extraction_id_attribute, getter=span_label)
        for attribute in (
            self.extraction_ids_attribute,
            self.facet_values_attribute,
            self.facet_known_values_attribute,
        ):
            if not Span.has_extension(attribute):
                Span.set_extension(attribute, default=None)

    @property
    def cfg(self):
        return {
            "target_span_getter": self.target_span_getter,
            "span_setter": self.span_setter,
            "query_attribute": self.query_attribute,
            "extraction_ids_attribute": self.extraction_ids_attribute,
            "extraction_id_attribute": self.extraction_id_attribute,
            "facet_values_attribute": self.facet_values_attribute,
            "facet_known_values_attribute": self.facet_known_values_attribute,
            "projection_size": self.projection_size,
            "extraction_threshold": self.extraction_threshold,
            "multi_label_threshold": self.multi_label_threshold,
        }

    def preprocess(self, doc: Doc, **kwargs) -> Dict[str, Any]:
        """Parse one report query bundle and prepare its shared encoding"""

        queries = getattr(doc._, self.query_attribute)
        extractions = queries["extractions"]
        classifications = queries.get("classifications", [])
        extraction_ids = [item["id"] for item in extractions]
        if not extractions or len(extraction_ids) != len(set(extraction_ids)):
            raise ValueError("Extraction ids must be unique and non empty")

        prompts = [f"{EXTRACTION_MARKER} {item['prompt']}" for item in extractions]
        groups = []
        seen_facets = set()
        for classification in classifications:
            facet = classification["facet"]
            cardinality = classification["cardinality"]
            options = classification["options"]
            value_ids = [option["value_id"] for option in options]
            if facet in seen_facets or cardinality not in ("single", "multi"):
                raise ValueError("Classification facets and cardinalities are invalid")
            if not options or len(value_ids) != len(set(value_ids)):
                raise ValueError(
                    "Classification option ids must be unique and non empty"
                )
            seen_facets.add(facet)
            option_prompt_indices = list(
                range(len(prompts), len(prompts) + len(options))
            )
            prompts.extend(
                f"{CLASSIFICATION_MARKER} {option['prompt']}" for option in options
            )
            groups.append(
                {
                    "facet": facet,
                    "cardinality": cardinality,
                    "value_ids": value_ids,
                    "option_prompt_indices": option_prompt_indices,
                }
            )

        return {
            "embedding": self.embedding.preprocess(
                doc,
                contexts=[doc[:]],
                segments=[prompts],
                **kwargs,
            ),
            "length": len(doc),
            "num_prompts": len(prompts),
            "extraction_ids": extraction_ids,
            "extraction_prompt_indices": list(range(len(extractions))),
            "classification_groups": groups,
            "stats": {"multi_query_qa_words": len(doc)},
        }

    def preprocess_supervised(self, doc: Doc, **kwargs) -> Dict[str, Any]:
        """Attach exact gold union and masked pair targets to one prepared report"""

        prep = self.preprocess(doc, **kwargs)
        spans = list(get_spans(doc, self.target_span_getter))
        if len({(span.start, span.end) for span in spans}) != len(spans):
            raise ValueError("Gold spans must have unique exact offsets")
        extraction_to_idx = {
            extraction_id: idx
            for idx, extraction_id in enumerate(prep["extraction_ids"])
        }
        extraction_targets = []
        for span in spans:
            extraction_ids = getattr(span._, self.extraction_ids_attribute) or ()
            unknown_ids = set(extraction_ids) - set(extraction_to_idx)
            if unknown_ids:
                raise ValueError("Gold extraction ids must occur in Doc queries")
            extraction_targets.append(
                [extraction_to_idx[extraction_id] for extraction_id in extraction_ids]
            )

        groups = []
        for group in prep["classification_groups"]:
            value_to_idx = {
                value_id: idx for idx, value_id in enumerate(group["value_ids"])
            }
            targets = []
            for span in spans:
                facet_values = getattr(span._, self.facet_values_attribute) or {}
                facet_known_values = (
                    getattr(span._, self.facet_known_values_attribute) or {}
                )
                values = facet_values.get(group["facet"])
                known_values = facet_known_values.get(group["facet"])
                if values is None or known_values is None:
                    targets.append(
                        -100
                        if group["cardinality"] == "single"
                        else [-100] * len(value_to_idx)
                    )
                    continue
                unknown_values = set(values) - set(value_to_idx)
                if unknown_values:
                    raise ValueError("Gold facet values must occur in Doc queries")
                if group["cardinality"] == "single":
                    if len(values) != 1:
                        raise ValueError("Known single facets need one value")
                    targets.append(value_to_idx[values[0]])
                else:
                    targets.append(
                        [
                            int(value_id in values)
                            if value_id in known_values
                            else -100
                            for value_id in group["value_ids"]
                        ]
                    )
            groups.append({**group, "targets": targets})

        return {
            **prep,
            "classification_groups": groups,
            "gold_spans": [[span.start, span.end] for span in spans],
            "gold_extraction_indices": extraction_targets,
            "stats": {
                **prep["stats"],
                "multi_query_qa_gold_spans": len(spans),
            },
        }

    def collate(self, batch: Dict[str, Sequence[Any]]) -> MultiQueryQABatchInput:
        """Pack ragged reports queries candidates and targets for vectorized scoring"""

        lengths = list(batch["length"])
        prompt_offsets = [0, *accumulate(batch["num_prompts"])][:-1]
        extraction_counts = [len(values) for values in batch["extraction_ids"]]
        extraction_offsets = [0, *accumulate(extraction_counts)][:-1]
        group_counts = [len(values) for values in batch["classification_groups"]]
        group_offsets = [0, *accumulate(group_counts)][:-1]

        extraction_context_indices = [
            sample_idx
            for sample_idx, values in enumerate(batch["extraction_ids"])
            for _ in values
        ]
        extraction_prompt_indices = [
            prompt_offsets[sample_idx] + prompt_idx
            for sample_idx, indices in enumerate(batch["extraction_prompt_indices"])
            for prompt_idx in indices
        ]
        extraction_ids = [
            extraction_id
            for sample_ids in batch["extraction_ids"]
            for extraction_id in sample_ids
        ]

        group_contexts = []
        group_cardinalities = []
        group_option_counts = []
        option_prompt_indices = []
        facets = []
        value_ids = []
        for sample_idx, sample_groups in enumerate(batch["classification_groups"]):
            for group in sample_groups:
                group_contexts.append(sample_idx)
                group_cardinalities.append(group["cardinality"] == "single")
                group_option_counts.append(len(group["option_prompt_indices"]))
                option_prompt_indices.extend(
                    prompt_offsets[sample_idx] + idx
                    for idx in group["option_prompt_indices"]
                )
                facets.append(group["facet"])
                value_ids.append(group["value_ids"])

        collated: MultiQueryQABatchInput = {
            "embedding": self.embedding.collate(batch["embedding"]),
            "lengths": torch.as_tensor(lengths, dtype=torch.long),
            "word_offsets": torch.as_tensor(
                [0, *accumulate(lengths)], dtype=torch.long
            ),
            "extraction_context_indices": torch.as_tensor(
                extraction_context_indices, dtype=torch.long
            ),
            "extraction_prompt_indices": torch.as_tensor(
                extraction_prompt_indices, dtype=torch.long
            ),
            "extraction_ids": extraction_ids,
            "classification_group_contexts": torch.as_tensor(
                group_contexts, dtype=torch.long
            ),
            "classification_group_cardinalities": torch.as_tensor(
                group_cardinalities, dtype=torch.bool
            ),
            "classification_group_option_counts": torch.as_tensor(
                group_option_counts, dtype=torch.long
            ),
            "classification_option_prompt_indices": torch.as_tensor(
                option_prompt_indices, dtype=torch.long
            ),
            "classification_facets": facets,
            "classification_value_ids": value_ids,
            "stats": {
                key: sum(values)
                for key, values in batch["stats"].items()
                if not key.startswith("__")
            },
        }

        if "gold_spans" not in batch:
            return collated

        gold_counts = [len(values) for values in batch["gold_spans"]]
        gold_offsets = [0, *accumulate(gold_counts)][:-1]
        gold_spans = torch.as_tensor(
            [
                [sample_idx, begin, end]
                for sample_idx, spans in enumerate(batch["gold_spans"])
                for begin, end in spans
            ],
            dtype=torch.long,
        ).reshape(-1, 3)
        gold_tags, conflicts = make_union_tags(
            gold_spans,
            len(lengths),
            max(lengths, default=0),
        )
        collated["gold_tags"] = gold_tags
        collated["gold_spans"] = gold_spans
        collated["gold_extraction_pairs"] = torch.as_tensor(
            [
                [
                    gold_offsets[sample_idx] + candidate_idx,
                    extraction_offsets[sample_idx] + extraction_idx,
                ]
                for sample_idx, candidate_targets in enumerate(
                    batch["gold_extraction_indices"]
                )
                for candidate_idx, extraction_indices in enumerate(candidate_targets)
                for extraction_idx in extraction_indices
            ],
            dtype=torch.long,
        ).reshape(-1, 2)

        gold_group_candidate_indices = []
        gold_group_indices = []
        gold_single_targets = []
        gold_option_group_pair_indices = []
        gold_option_offsets = []
        gold_multi_targets = []
        for sample_idx, sample_groups in enumerate(batch["classification_groups"]):
            for candidate_idx in range(gold_counts[sample_idx]):
                for local_group_idx, group in enumerate(sample_groups):
                    group_pair_idx = len(gold_group_indices)
                    gold_group_candidate_indices.append(
                        gold_offsets[sample_idx] + candidate_idx
                    )
                    gold_group_indices.append(
                        group_offsets[sample_idx] + local_group_idx
                    )
                    target = group["targets"][candidate_idx]
                    gold_single_targets.append(
                        target if group["cardinality"] == "single" else -100
                    )
                    option_count = len(group["option_prompt_indices"])
                    gold_option_group_pair_indices.extend(
                        [group_pair_idx] * option_count
                    )
                    gold_option_offsets.extend(range(option_count))
                    gold_multi_targets.extend(
                        target
                        if group["cardinality"] == "multi"
                        else [-100] * option_count
                    )

        collated["gold_group_candidate_indices"] = torch.as_tensor(
            gold_group_candidate_indices, dtype=torch.long
        )
        collated["gold_group_indices"] = torch.as_tensor(
            gold_group_indices, dtype=torch.long
        )
        collated["gold_single_targets"] = torch.as_tensor(
            gold_single_targets, dtype=torch.long
        )
        collated["gold_option_group_pair_indices"] = torch.as_tensor(
            gold_option_group_pair_indices, dtype=torch.long
        )
        collated["gold_option_offsets"] = torch.as_tensor(
            gold_option_offsets, dtype=torch.long
        )
        collated["gold_multi_targets"] = torch.as_tensor(
            gold_multi_targets, dtype=torch.float
        )
        collated["stats"]["multi_query_qa_crf_projection_conflicts"] = conflicts
        return collated

    def forward(self, batch: MultiQueryQABatchInput) -> MultiQueryQABatchOutput:
        """Score all packed queries from one candidate path"""

        embedding_output = self.embedding(batch["embedding"])
        folded_words = embedding_output["embeddings"]
        dense_words = folded_words.refold("context", "word")
        words = folded_words.refold("word").as_tensor()
        prompt_embeddings = embedding_output["segment_embeddings"].refold(
            "segment", "segment_token"
        )

        emissions = self.head.candidate_linear(dense_words.as_tensor())
        word_mask = dense_words.mask
        tags = self.head.crf.decode(emissions, word_mask)
        decoded_spans = self.head.crf.tags_to_spans(tags.unsqueeze(-1))[:, :3]
        supervised = "gold_spans" in batch
        # Dynamic candidate lengths cannot enter FoldedTensor without CPU length lists
        # Explicit context ids keep this union packed for every downstream scorer
        if supervised:
            candidates, inverse = torch.unique(
                torch.cat((batch["gold_spans"], decoded_spans)),
                dim=0,
                return_inverse=True,
            )
            gold_candidate_indices = inverse[: len(batch["gold_spans"])]
            candidate_loss = self.head.crf(
                emissions,
                word_mask,
                F.one_hot(batch["gold_tags"], 5).bool(),
            ).sum() / word_mask.sum().clamp_min(1)
        else:
            candidates = decoded_spans
            gold_candidate_indices = candidates.new_empty(0)
            candidate_loss = None

        span_features = self.head.encode_spans(
            words,
            candidates,
            batch["word_offsets"],
        )
        query_features = self.head.encode_queries(
            prompt_embeddings.as_tensor(),
            prompt_embeddings.mask,
        )
        candidate_contexts = candidates[:, 0]
        num_contexts = len(batch["lengths"])

        (
            extraction_pair_candidates,
            extraction_pair_queries,
            extraction_pair_starts,
            extraction_starts,
        ) = pair_by_context(
            candidate_contexts,
            batch["extraction_context_indices"],
            num_contexts,
        )
        extraction_scores = self.head.score_pairs(
            span_features,
            query_features,
            extraction_pair_candidates,
            batch["extraction_prompt_indices"][extraction_pair_queries],
        )

        (
            classification_pair_candidates,
            classification_pair_groups,
            classification_pair_starts,
            classification_group_starts,
        ) = pair_by_context(
            candidate_contexts,
            batch["classification_group_contexts"],
            num_contexts,
        )
        pair_option_counts = batch["classification_group_option_counts"][
            classification_pair_groups
        ]
        classification_score_pairs = torch.arange(
            len(classification_pair_groups), device=words.device
        ).repeat_interleave(pair_option_counts)
        classification_pair_option_starts = (
            pair_option_counts.cumsum(0) - pair_option_counts
        )
        score_option_offsets = torch.arange(
            len(classification_score_pairs), device=words.device
        ) - classification_pair_option_starts.repeat_interleave(pair_option_counts)
        classification_group_option_starts = (
            batch["classification_group_option_counts"].cumsum(0)
            - batch["classification_group_option_counts"]
        )
        classification_score_options = (
            classification_group_option_starts[
                classification_pair_groups[classification_score_pairs]
            ]
            + score_option_offsets
        )
        classification_scores = self.head.score_pairs(
            span_features,
            query_features,
            classification_pair_candidates[classification_score_pairs],
            batch["classification_option_prompt_indices"][classification_score_options],
        )

        pair_max = torch.full(
            (len(classification_pair_groups),),
            -torch.inf,
            device=words.device,
        ).scatter_reduce(
            0,
            classification_score_pairs,
            classification_scores,
            reduce="amax",
            include_self=True,
        )
        pair_sum = torch.zeros(
            len(classification_pair_groups), device=words.device
        ).scatter_add(
            0,
            classification_score_pairs,
            torch.exp(classification_scores - pair_max[classification_score_pairs]),
        )
        pair_logsumexp = pair_max + pair_sum.log()

        if supervised:
            extraction_targets = torch.zeros_like(extraction_scores)
            gold_extraction_pairs = batch["gold_extraction_pairs"]
            positive_candidates = gold_candidate_indices[gold_extraction_pairs[:, 0]]
            positive_contexts = candidate_contexts[positive_candidates]
            positive_pair_indices = (
                extraction_pair_starts[positive_candidates]
                + gold_extraction_pairs[:, 1]
                - extraction_starts[positive_contexts]
            )
            extraction_targets[positive_pair_indices] = 1
            extraction_loss = F.binary_cross_entropy_with_logits(
                extraction_scores,
                extraction_targets,
                reduction="sum",
            ) / max(len(extraction_scores), 1)

            gold_group_candidates = gold_candidate_indices[
                batch["gold_group_candidate_indices"]
            ]
            gold_group_contexts = candidate_contexts[gold_group_candidates]
            gold_runtime_pairs = (
                classification_pair_starts[gold_group_candidates]
                + batch["gold_group_indices"]
                - classification_group_starts[gold_group_contexts]
            )
            single_targets = torch.full(
                (len(classification_pair_groups),),
                -100,
                dtype=torch.long,
                device=words.device,
            ).index_copy(
                0,
                gold_runtime_pairs,
                batch["gold_single_targets"],
            )
            multi_targets = torch.full_like(classification_scores, -100).index_copy(
                0,
                classification_pair_option_starts[
                    gold_runtime_pairs[batch["gold_option_group_pair_indices"]]
                ]
                + batch["gold_option_offsets"],
                batch["gold_multi_targets"],
            )
            single_known = batch["classification_group_cardinalities"][
                classification_pair_groups
            ] & (single_targets != -100)
            target_scores = classification_scores[
                classification_pair_option_starts + single_targets.clamp_min(0)
            ]
            single_loss = (pair_logsumexp - target_scores)[single_known].sum()
            multi_known = ~batch["classification_group_cardinalities"][
                classification_pair_groups[classification_score_pairs]
            ] & (multi_targets != -100)
            multi_losses = (
                F.softplus(classification_scores)
                - multi_targets.clamp_min(0) * classification_scores
            )
            classification_count = single_known.sum() + multi_known.sum()
            classification_loss = (
                single_loss + multi_losses[multi_known].sum()
            ) / classification_count.clamp_min(1)
            loss = candidate_loss + extraction_loss + classification_loss
            extraction_selected = None
            classification_selected = None
        else:
            extraction_loss = None
            classification_loss = None
            loss = None
            extraction_selected = (
                extraction_scores.sigmoid() >= self.extraction_threshold
            )
            best_option_offsets = torch.full(
                (len(classification_pair_groups),),
                len(batch["classification_option_prompt_indices"]),
                dtype=torch.long,
                device=words.device,
            ).scatter_reduce(
                0,
                classification_score_pairs,
                torch.where(
                    classification_scores == pair_max[classification_score_pairs],
                    score_option_offsets,
                    len(batch["classification_option_prompt_indices"]),
                ),
                reduce="amin",
                include_self=True,
            )
            single_selected = (
                score_option_offsets == best_option_offsets[classification_score_pairs]
            )
            classification_selected = torch.where(
                batch["classification_group_cardinalities"][
                    classification_pair_groups[classification_score_pairs]
                ],
                single_selected,
                classification_scores.sigmoid() >= self.multi_label_threshold,
            )

        return {
            "loss": loss,
            "candidate_loss": candidate_loss,
            "extraction_loss": extraction_loss,
            "classification_loss": classification_loss,
            "tags": tags,
            "candidate_spans": candidates,
            "extraction_ids": batch["extraction_ids"],
            "extraction_pair_candidates": extraction_pair_candidates,
            "extraction_pair_queries": extraction_pair_queries,
            "extraction_selected": extraction_selected,
            "classification_pair_candidates": classification_pair_candidates,
            "classification_pair_groups": classification_pair_groups,
            "classification_score_pairs": classification_score_pairs,
            "classification_score_options": classification_score_options,
            "classification_selected": classification_selected,
            "classification_group_option_starts": classification_group_option_starts,
            "classification_facets": batch["classification_facets"],
            "classification_value_ids": batch["classification_value_ids"],
        }

    def postprocess(
        self,
        docs: Sequence[Doc],
        results: MultiQueryQABatchOutput,
        inputs: List[Dict[str, Any]],
    ) -> Sequence[Doc]:
        """Attach one duplicate offset span for every selected extraction query"""

        facet_values = [{} for _ in range(len(results["candidate_spans"]))]
        facets = results["classification_facets"]
        for candidate_idx, group_idx in zip(
            results["classification_pair_candidates"].cpu().tolist(),
            results["classification_pair_groups"].cpu().tolist(),
        ):
            facet_values[candidate_idx][facets[group_idx]] = []
        selected_options = torch.nonzero(results["classification_selected"]).flatten()
        selected_pairs = results["classification_score_pairs"][selected_options]
        for candidate_idx, group_idx, option_idx in zip(
            results["classification_pair_candidates"][selected_pairs].cpu().tolist(),
            results["classification_pair_groups"][selected_pairs].cpu().tolist(),
            results["classification_score_options"][selected_options].cpu().tolist(),
        ):
            local_option_idx = (
                option_idx
                - results["classification_group_option_starts"][group_idx].item()
            )
            facet_values[candidate_idx][facets[group_idx]].append(
                results["classification_value_ids"][group_idx][local_option_idx]
            )

        spans_by_doc = defaultdict(list)
        selected_extractions = torch.nonzero(results["extraction_selected"]).flatten()
        for candidate_idx, extraction_idx in zip(
            results["extraction_pair_candidates"][selected_extractions].cpu().tolist(),
            results["extraction_pair_queries"][selected_extractions].cpu().tolist(),
        ):
            sample_idx, begin, end = results["candidate_spans"][candidate_idx].tolist()
            extraction_id = results["extraction_ids"][extraction_idx]
            span = Span(docs[sample_idx], begin, end, label=extraction_id)
            setattr(
                span._,
                self.facet_values_attribute,
                {
                    facet: tuple(values)
                    for facet, values in facet_values[candidate_idx].items()
                },
            )
            spans_by_doc[docs[sample_idx]].append(span)
        for doc in docs:
            self.set_spans(doc, spans_by_doc.get(doc, []))
        return docs
