from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, OrderedDict, Tuple

import torch
from loguru import logger
from spacy import registry
from spacy.tokens import Doc
from thinc.layers import chain
from thinc.model import Model
from thinc.shims import PyTorchShim
from thinc.types import ArgsKwargs, Floats1d, Floats2d, Ints2d
from thinc.util import (
    convert_recursive,
    is_torch_array,
    is_xp_array,
    torch2xp,
    xp2torch,
)

from edsnlp.models.torch.crf import IMPOSSIBLE, MultiLabelBIOULDecoder


class CRFMode(Enum):
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


class NestedNERModule(torch.nn.Module):
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
        super().__init__()

        self.mode = mode

        assert mode in (CRFMode.independent, CRFMode.joint, CRFMode.marginal)
        self.crf = MultiLabelBIOULDecoder(1, learnable_transitions=False)
        self.n_labels = n_labels
        self.input_size = input_size

        self.classifier = None

    def load_state_dict(
        self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True
    ):
        """
        Loads the model inplace from a dumped `state_dict` object

        Parameters
        ----------
        state_dict: OrderedDict[str, torch.Tensor]
        strict: bool
        """
        self.n_labels = state_dict["classifier.weight"].shape[0] // self.crf.num_tags
        self.input_size = state_dict["classifier.weight"].shape[1]
        self.initialize()
        super().load_state_dict(state_dict, strict)

    def set_n_labels(self, n_labels):
        """
        Sets the number of labels. To instanciate the linear layer, we need to
        call the `initialize` method.

        Parameters
        ----------
        n_labels: int
            Number of different labels predicted by this module
        """
        self.n_labels = n_labels

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
    ):
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
            if self.mode == CRFMode.joint:
                loss = self.crf(
                    crf_logits,
                    crf_mask,
                    crf_target,
                )
            elif self.mode == CRFMode.independent:
                loss = (
                    -crf_logits.log_softmax(-1)
                    .masked_fill(~crf_target, IMPOSSIBLE)
                    .logsumexp(-1)[crf_mask]
                    .sum()
                )
            elif self.mode == CRFMode.marginal:
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


def xp2torch_for_ner_crf(model, X):
    main = xp2torch(X[0], requires_grad=True)
    rest = convert_recursive(is_xp_array, lambda x: xp2torch(x), X[1:])

    def reverse_conversion(dXtorch):
        dX = torch2xp(dXtorch.args[0])
        return dX

    return (main, *rest), reverse_conversion


def list_to_unsorted_padded():
    """
    Make an item padder model that does not sort them.
    We do not sort the items since:
    1. we don't need that
    2. it messes up the indices to keep track of the predictions made by the model
    """

    def forward(model, items, is_train=False):
        res = model.ops.pad(items)

        def backprop(d_padded):
            return [
                d_padded_row[: len(d_item)]
                for d_item, d_padded_row in zip(items, d_padded)
            ]

        return res, backprop

    return Model(
        "list_to_unsorted_padded",
        forward,
        dims={"nI": None, "nO": None},
    )


@registry.layers("eds.nested_ner_model.v1")
def create_nested_ner_model(
    tok2vec: Model[List[Doc], List[Floats2d]],
    mode: CRFMode,
    n_labels: int = None,
) -> Model[
    # inputs (docs, gold, is_predict)
    Tuple[Iterable[Doc], Optional[Ints2d], Optional[bool]],
    # outputs (loss + spans)
    Tuple[Floats1d, Ints2d],
]:
    padded_tok2vec = chain(tok2vec, list_to_unsorted_padded())
    pt_model = NestedNERModule(
        input_size=None,  # will be set later during initialization
        n_labels=n_labels,  # will likely be set later during initialization
        mode=mode,
    )
    return Model(
        "pytorch",
        nested_ner_forward,
        attrs={
            "set_n_labels": pt_model.set_n_labels,
            "pt_model": pt_model,
        },
        layers=[padded_tok2vec],
        shims=[PyTorchShim(pt_model)],
        refs={"tok2vec": padded_tok2vec},
        dims={"nI": None, "nO": None},
        init=instance_init,
    )


def nested_ner_forward(
    model: Model,
    X: Tuple[Iterable[Doc], Ints2d, bool],
    is_train: bool = False,
) -> Tuple[Tuple[Floats1d, Ints2d], Callable[[Floats1d], Any]]:
    """
    Run the stacked CRF pytorch model to train / run a nested NER model

    Parameters
    ----------
    model: Model
    X: Tuple[Iterable[Doc], Ints2d, bool]
    is_train: bool

    Returns
    -------
    Tuple[Tuple[Floats1d, Ints2d], Callable[Floats1d, Any]]
    """
    [docs, spans, is_predict] = X
    tok2vec: Model[List[Doc], List[Floats2d]] = model.get_ref("tok2vec")
    embeds, bp_embeds = tok2vec(docs, is_train=is_train)

    ##################################################
    # Prepare the torch nested ner crf module inputs #
    ##################################################
    additional_outputs = {"spans": None}
    # Convert input from numpy/cupy to torch
    (torch_embeds, *torch_rest), get_d_embeds = xp2torch_for_ner_crf(
        model, (embeds, spans)
    )
    # Prepare token mask from docs' lengths
    torch_mask = (
        torch.arange(embeds.shape[1], device=torch_embeds.device)
        < torch.tensor([len(d) for d in docs], device=torch_embeds.device)[:, None]
    )

    #################
    # Run the model #
    #################
    loss_torch, torch_backprop = model.shims[0](
        ArgsKwargs(
            (torch_embeds, torch_mask, *torch_rest),
            {
                "additional_outputs": additional_outputs,
                "is_train": is_train,
                "is_predict": is_predict,
            },
        ),
        is_train,
    )

    ####################################
    # Postprocess the module's outputs #
    ####################################
    loss = torch2xp(loss_torch) if loss_torch is not None else None
    additional_outputs = convert_recursive(is_torch_array, torch2xp, additional_outputs)
    spans = additional_outputs["spans"]

    def backprop(d_loss: Floats1d) -> Any:
        d_loss_torch = ArgsKwargs(
            args=((loss_torch,),), kwargs={"grad_tensors": xp2torch(d_loss)}
        )
        d_embeds_torch = torch_backprop(d_loss_torch)
        d_embeds = get_d_embeds(d_embeds_torch)
        d_docs = bp_embeds(d_embeds)
        return d_docs

    return (loss, spans), backprop


def instance_init(model: Model, X: List[Doc] = None, Y: Ints2d = None) -> Model:
    """
    Initializes the model by setting the input size of the model layers and the number
    of predicted labels

    Parameters
    ----------
    model: Model
        Nested NER thinc model
    X: List[Doc]
        list of documents on which we apply the tok2vec layer
    Y: Ints2d
        Unused gold spans

    Returns
    -------

    """
    tok2vec = model.get_ref("tok2vec")
    if X is not None:
        tok2vec.initialize(X)

    pt_model = model.attrs["pt_model"]
    pt_model.input_size = tok2vec.get_dim("nO")
    pt_model.initialize()
    model.set_dim("nI", pt_model.input_size)

    return model
