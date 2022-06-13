from typing import Any, Iterable, List, OrderedDict, Tuple

import numpy as np
import torch
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

from edsnlp.models.torch.crf import IMPOSSIBLE, BIOULDecoder


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
    def __init__(self, input_size, n_labels, mode="joint"):
        super().__init__()

        self.input_size = input_size
        self.mode = mode
        assert mode in ("independent", "joint", "marginal")
        self.crf = BIOULDecoder(1, learnable_transitions=False)
        self.num_tags = 0
        self.n_labels = 0
        self.classifier = None
        if n_labels is not None:
            self.set_n_labels(n_labels)
        else:
            self.set_n_labels(1)

    def load_state_dict(
        self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True
    ):
        self.set_n_labels(state_dict["classifier.weight"].shape[0] // self.crf.num_tags)
        super().load_state_dict(state_dict, strict)

    def set_n_labels(self, n_labels):
        if n_labels is None:
            return

        self.num_tags = n_labels * self.crf.num_tags
        self.n_labels = n_labels
        self.classifier = torch.nn.Linear(self.input_size, self.num_tags)

    def forward(
        self,
        embeds,
        mask,
        spans=None,
        additional_outputs=None,
        is_train=False,
        is_predict=False,
    ):
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
            if self.mode == "independent":
                loss = self.crf(
                    crf_logits,
                    crf_mask,
                    crf_target,
                )
            elif self.mode == "joint":
                loss = (
                    -crf_logits.log_softmax(-1)
                    .masked_fill(~crf_target, IMPOSSIBLE)
                    .logsumexp(-1)[crf_mask]
                    .sum()
                )
            elif self.mode == "marginal":
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
                raise
            loss = loss.sum() / 100.0
        if is_predict:
            pred_tags = self.crf.decode(crf_logits, crf_mask).reshape(
                n_samples, self.n_labels, n_tokens
            )
            pred_spans = self.crf.tags_to_spans(pred_tags)
            additional_outputs["spans"] = pred_spans
        return loss


def convert_ner_crf_inputs(model, X):
    main = xp2torch(X[0], requires_grad=True)
    rest = convert_recursive(is_xp_array, lambda x: xp2torch(x), X[1:])

    def reverse_conversion(dXtorch):
        dX = torch2xp(dXtorch.args[0])
        return dX

    return (main, *rest), reverse_conversion


def list_to_unsorted_padded():
    def forward(model, items, is_train=False):
        res = np.zeros(
            (len(items), max(map(len, items)), items[0].shape[-1]), dtype=items[0].dtype
        )
        for i, item in enumerate(items):
            res[i, : len(item)] = item

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
    mode: str,
    n_labels: int = None,
) -> Model[Any, Any]:
    tok2vec_size = tok2vec.get_dim("nO")
    padded_tok2vec = chain(tok2vec, list_to_unsorted_padded())
    pt_model = NestedNERModule(input_size=tok2vec_size, n_labels=n_labels, mode=mode)
    return Model(
        "pytorch",
        nested_ner_forward,
        attrs={
            "set_n_labels": pt_model.set_n_labels,
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
):
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
    tok2vec = model.get_ref("tok2vec")
    embeds, bp_embeds = tok2vec(docs, is_train=is_train)
    X = [embeds, spans]

    additional_outputs = {"spans": None}
    (torch_embeds, *torch_rest), get_d_embeds = convert_ner_crf_inputs(
        model, (embeds, spans)
    )
    torch_mask = (
        torch.arange(embeds.shape[1], device=torch_embeds.device)
        < torch.tensor([len(d) for d in docs], device=torch_embeds.device)[:, None]
    )
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
    loss = torch2xp(loss_torch) if loss_torch is not None else None
    additional_outputs = convert_recursive(is_torch_array, torch2xp, additional_outputs)

    def backprop(d_loss: Floats1d) -> Any:
        d_loss_torch = ArgsKwargs(
            args=((loss_torch,),), kwargs={"grad_tensors": xp2torch(d_loss)}
        )
        d_embeds_torch = torch_backprop(d_loss_torch)
        d_embeds = get_d_embeds(d_embeds_torch)
        d_docs = bp_embeds(d_embeds)
        return d_docs

    return (loss, additional_outputs["spans"]), backprop


def instance_init(model: Model, X: List[Doc] = None, Y: Ints2d = None) -> Model:
    tok2vec = model.get_ref("tok2vec")
    if X is not None:
        tok2vec.initialize(X)
    return model
