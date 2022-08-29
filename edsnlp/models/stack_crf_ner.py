from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from loguru import logger
from spacy import registry
from spacy.tokens import Doc
from thinc.model import Model
from thinc.types import Floats1d, Floats2d, Ints2d

from edsnlp.models.pytorch_wrapper import PytorchWrapperModule, wrap_pytorch_model
from edsnlp.models.torch.crf import IMPOSSIBLE, MultiLabelBIOULDecoder


class CRFMode(str, Enum):
    independent = "independent"
    joint = "joint"
    marginal = "marginal"


def repeat(t, n, dim, interleave=True):
    repeat_dim = dim + (1 if interleave else 0)
    return (
        t.unsqueeze(repeat_dim)
        .repeat_interleave(n, repeat_dim)
        .view(
            tuple(
                -1 if (i - dim + t.ndim) % t.ndim == 0 else s
                for i, s in enumerate(t.shape)
            )
        )
    )


def flatten_dim(t, dim, ndim=1):
    return t.reshape((*t.shape[:dim], -1, *t.shape[dim + 1 + ndim :]))


class StackedCRFNERModule(PytorchWrapperModule):
    def __init__(
        self,
        input_size: Optional[int] = None,
        n_labels: Optional[int] = None,
        mode: CRFMode = CRFMode.joint,
    ):
        """
        Nested NER CRF module

        Parameters
        ----------
        input_size: int
            Size of the input embeddings
        n_labels: int
            Number of labels predicted by the module
        mode: CRFMode
            Loss mode of the CRF
        """
        super().__init__(input_size, n_labels)

        self.cfg["mode"] = mode

        assert mode in (CRFMode.independent, CRFMode.joint, CRFMode.marginal)
        self.crf = MultiLabelBIOULDecoder(1, learnable_transitions=False)

        self.classifier = None

    def initialize(self):
        """
        Once the number of labels n_labels are known, this method
        initializes the torch linear layer.
        """
        num_tags = self.n_labels * self.crf.num_tags
        self.classifier = torch.nn.Linear(self.input_size, num_tags)

    def forward(
        self,
        embeds: torch.FloatTensor,
        mask: torch.BoolTensor,
        spans: Optional[torch.LongTensor] = None,
        additional_outputs: Dict[str, Any] = None,
        is_train: bool = False,
        is_predict: bool = False,
    ) -> Optional[torch.FloatTensor]:
        """
        Apply the nested ner module to the document embeddings to:
        - compute the loss
        - predict the spans
        non exclusively.
        If spans are predicted, they are assigned to the `additional_outputs`
        dictionary.

        Parameters
        ----------
        embeds: torch.FloatTensor
            Token embeddings to predict the tags from
        mask: torch.BoolTensor
            Mask of the sequences
        spans: Optional[torch.LongTensor]
            2d tensor of n_spans * (doc_idx, label_idx, begin, end)
        additional_outputs: Dict[str, Any]
            Additional outputs that should not / cannot be back-propped through
            (Thinc treats Pytorch models solely as derivable functions, but the CRF
            that we employ performs the best tag decoding function with Pytorch)
            This dict will contain the predicted 2d tensor of spans
        is_train: bool=False
            Are we training the model (defaults to True)
        is_predict: bool=False
            Are we predicting the model (defaults to False)

        Returns
        -------
        Optional[torch.FloatTensor]
            Optional 0d loss (shape = [1]) to train the model
        """
        n_samples, n_tokens = embeds.shape[:2]
        logits = self.classifier(embeds)
        crf_logits = flatten_dim(
            logits.view(n_samples, n_tokens, self.n_labels, self.crf.num_tags).permute(
                0, 2, 1, 3
            ),
            dim=0,
        )
        crf_mask = repeat(mask, self.n_labels, 0)
        loss = None
        if is_train:
            tags = self.crf.spans_to_tags(
                spans, n_samples=n_samples, n_tokens=n_tokens, n_labels=self.n_labels
            )
            crf_target = flatten_dim(
                torch.nn.functional.one_hot(tags, 5).bool()
                if len(tags.shape) == 3
                else tags,
                dim=0,
            )
            if self.cfg["mode"] == CRFMode.joint:
                loss = self.crf(
                    crf_logits,
                    crf_mask,
                    crf_target,
                )
            elif self.cfg["mode"] == CRFMode.independent:
                loss = (
                    -crf_logits.log_softmax(-1)
                    .masked_fill(~crf_target, IMPOSSIBLE)
                    .logsumexp(-1)[crf_mask]
                    .sum()
                )
            elif self.cfg["mode"] == CRFMode.marginal:
                crf_logits = self.crf.marginal(
                    crf_logits,
                    crf_mask,
                )
                loss = (
                    -crf_logits.log_softmax(-1)
                    .masked_fill(~crf_target, IMPOSSIBLE)
                    .logsumexp(-1)[crf_mask]
                    .sum()
                )
            if (loss > -IMPOSSIBLE).any():
                logger.warning(
                    "You likely have an impossible transition in your "
                    "training data NER tags, skipping this batch."
                )
                loss = torch.zeros(1, dtype=torch.float, device=embeds.device)
            loss = loss.sum().unsqueeze(0) / 100.0
        if is_predict:
            pred_tags = self.crf.decode(crf_logits, crf_mask).reshape(
                n_samples, self.n_labels, n_tokens
            )
            pred_spans = self.crf.tags_to_spans(pred_tags)
            additional_outputs["spans"] = pred_spans
        return loss


@registry.layers("eds.stack_crf_ner_model.v1")
def create_model(
    tok2vec: Model[List[Doc], List[Floats2d]],
    mode: CRFMode,
    n_labels: int = None,
) -> Model[
    Tuple[Iterable[Doc], Optional[Ints2d], Optional[bool]],
    Tuple[Floats1d, Ints2d],
]:
    return wrap_pytorch_model(  # noqa
        encoder=tok2vec,
        pt_model=StackedCRFNERModule(
            input_size=None,  # will be set later during initialization
            n_labels=n_labels,  # will likely be set later during initialization
            mode=mode,
        ),
    )
