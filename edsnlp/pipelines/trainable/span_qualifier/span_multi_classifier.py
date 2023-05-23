from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, OrderedDict, Tuple

import torch
from spacy import registry
from spacy.tokens import Doc
from thinc.model import Model
from thinc.types import Floats1d, Floats2d, Ints2d

from edsnlp.pipelines.trainable.pytorch_wrapper import (
    PytorchWrapperModule,
    wrap_pytorch_model,
)


class ProjectionMode(str, Enum):
    dot = "dot"


class PoolerMode(str, Enum):
    max = "max"
    sum = "sum"
    mean = "mean"


class SpanMultiClassifier(PytorchWrapperModule):
    def __init__(
        self,
        input_size: Optional[int] = None,
        n_labels: Optional[int] = None,
        pooler_mode: PoolerMode = "max",
        projection_mode: ProjectionMode = "dot",
    ):
        """
        Pytorch module for constrained multi-label & multi-class span classification

        Parameters
        ----------
        input_size: int
            Size of the input embeddings
        n_labels: int
            Number of labels predicted by the module
        pooler_mode: PoolerMode
            How embeddings are aggregated
        projection_mode: ProjectionMode
            How embeddings converted into logits
        """
        super().__init__(input_size, n_labels)

        self.cfg["projection_mode"] = projection_mode
        self.cfg["pooler_mode"] = pooler_mode

        if projection_mode != "dot":
            raise Exception(
                "Only scalar product is supported " "for label classification."
            )

        self.groups_indices = None
        self.classifier = None

    def initialize(self):
        """
        Once the number of labels n_labels are known, this method
        initializes the torch linear layer.
        """
        if self.cfg["projection_mode"] == "dot":
            self.classifier = torch.nn.Linear(self.input_size, self.n_labels)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict()

        sd["groups_indices"] = self.groups_indices
        sd["combinations"] = list(self.combinations)
        return sd

    def load_state_dict(
        self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True
    ):
        if state_dict.get("combinations", None) is not None:
            self.set_label_groups(
                groups_combinations=state_dict.pop("combinations"),
                groups_indices=state_dict.pop("groups_indices"),
            )

        super().load_state_dict(state_dict, strict)

    def set_label_groups(
        self,
        groups_combinations,
        groups_indices,
    ):
        """
        Set the label groups matrices.
        """

        # To make the buffers discoverable by pytorch (for device moving operations),
        # we need to register them as buffer, and then we can group them in a
        # single list of tensors
        self.groups_indices = groups_indices
        for i, group_combinations in enumerate(groups_combinations):
            # n_combinations_in_group * n_labels_in_group
            self.register_buffer(
                f"combinations_{i}",
                torch.as_tensor(group_combinations, dtype=torch.bool),
            )

    @property
    def combinations(self):
        for i in range(len(self.groups_indices)):
            yield getattr(self, f"combinations_{i}")

    def forward(
        self,
        embeds: torch.FloatTensor,
        mask: torch.BoolTensor,
        spans: Optional[torch.LongTensor],
        targets: Optional[torch.LongTensor],
        additional_outputs: Dict[str, Any] = None,
        is_train: bool = False,
        is_predict: bool = False,
    ) -> Optional[torch.FloatTensor]:
        """
        Apply the span classifier module to the document embeddings and given spans to:
        - compute the loss
        - and/or predict the labels of spans
        If labels are predicted, they are assigned to the `additional_outputs`
        dictionary.

        Parameters
        ----------
        embeds: torch.FloatTensor
            Token embeddings to predict the tags from
        mask: torch.BoolTensor
            Mask of the sequences
        spans: Optional[torch.LongTensor]
            2d tensor of n_spans * (doc_idx, ner_label_idx, begin, end)
        targets: Optional[List[torch.LongTensor]]
            list of 2d tensor of n_spans * n_combinations (1 hot)
        additional_outputs: Dict[str, Any]
            Additional outputs that should not / cannot be back-propped through
            This dict will contain the predicted 2d tensor of labels
        is_train: bool=False
            Are we training the model (defaults to True)
        is_predict: bool=False
            Are we predicting the model (defaults to False)

        Returns
        -------
        Optional[torch.FloatTensor]
            Optional 0d loss (shape = [1]) to train the model
        """
        n_samples, n_words = embeds.shape[:2]
        device = embeds.device
        (sample_idx, span_begins, span_ends) = spans.unbind(1)
        if len(span_begins) == 0:
            loss = None
            if is_train:
                loss = embeds.sum().unsqueeze(0) * 0
            else:
                additional_outputs["labels"] = torch.zeros(
                    0, self.n_labels, device=embeds.device, dtype=torch.int
                )
            return loss

        flat_begins = n_words * sample_idx + span_begins
        flat_ends = n_words * sample_idx + span_ends
        flat_embeds = embeds.view(-1, embeds.shape[-1])
        flat_indices = torch.cat(
            [
                torch.arange(b, e, device=device)
                for b, e in zip(flat_begins.cpu().tolist(), flat_ends.cpu().tolist())
            ]
        ).to(embeds.device)
        offsets = (flat_ends - flat_begins).cumsum(0).roll(1)
        offsets[0] = 0
        span_embeds = torch.nn.functional.embedding_bag(
            input=flat_indices,
            weight=flat_embeds,
            offsets=offsets,
            mode=self.cfg["pooler_mode"],
        )

        scores = self.classifier(span_embeds)

        groups_combinations_scores = [
            # ([e]ntities * [b]indings) * ([c]ombinations * [b]indings)
            torch.einsum("eb,cb->ec", scores[:, grp_ids], grp_combinations.float())
            for grp_combinations, grp_ids in zip(self.combinations, self.groups_indices)
        ]  # -> list of ([e]ntities * [c]ombinations)

        loss = None
        if is_train:
            loss = sum(
                [
                    -grp_combinations_scores.log_softmax(-1)
                    .masked_fill(~grp_gold_combinations.to(device).bool(), 0)
                    .sum()
                    for grp_combinations_scores, grp_gold_combinations in zip(
                        groups_combinations_scores, targets
                    )
                ]
            )
            loss = loss.unsqueeze(0)  # for the thinc-pytorch shim
        if is_predict:
            pred = torch.cat(
                [
                    group_combinations[group_scores.argmax(-1)]
                    for group_scores, group_combinations in zip(
                        groups_combinations_scores, self.combinations
                    )
                ],
                dim=-1,
            )
            additional_outputs["labels"] = pred.int()
        return loss


@registry.layers("eds.span_multi_classifier.v1")
def create_model(
    tok2vec: Model[List[Doc], List[Floats2d]],
    projection_mode: ProjectionMode = ProjectionMode.dot,
    pooler_mode: PoolerMode = PoolerMode.max,
    n_labels: int = None,
) -> Model[
    Tuple[Iterable[Doc], Optional[Ints2d], Optional[bool]],
    Tuple[Floats1d, Ints2d],
]:
    return wrap_pytorch_model(  # noqa
        encoder=tok2vec,
        pt_model=SpanMultiClassifier(
            input_size=None,  # will be set later during initialization
            n_labels=n_labels,  # will likely be set later during initialization
            projection_mode=projection_mode,
            pooler_mode=pooler_mode,
        ),
        attrs=[
            "set_n_labels",
            "set_label_groups",
        ],
    )
