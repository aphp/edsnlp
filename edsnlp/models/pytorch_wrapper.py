import typing
from typing import Any, Callable, Iterable, List, Optional, OrderedDict, Tuple

import torch
from spacy.tokens import Doc
from thinc.model import Model
from thinc.shims import PyTorchShim
from thinc.types import ArgsKwargs, Floats1d, Floats2d, Ints2d
from thinc.util import (
    convert_recursive,
    get_torch_default_device,
    is_torch_array,
    is_xp_array,
    torch2xp,
    xp2torch,
)

PredT = typing.TypeVar("PredT")


class PytorchWrapperModule(torch.nn.Module):
    def __init__(
        self,
        input_size: Optional[int] = None,
        n_labels: Optional[int] = None,
    ):
        """
        Pytorch wrapping module for Spacy.
        Models that expect to be wrapped with
        [wrap_pytorch_model][edsnlp.models.pytorch_wrapper.wrap_pytorch_model]
        should inherit from this module.

        Parameters
        ----------
        input_size: int
            Size of the input embeddings
        n_labels: int
            Number of labels predicted by the module
        """
        super().__init__()

        self.cfg = {"n_labels": n_labels, "input_size": input_size}

    @property
    def n_labels(self):
        return self.cfg["n_labels"]

    @property
    def input_size(self):
        return self.cfg["input_size"]

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
        self.cfg = state_dict.pop("cfg")
        self.initialize()
        super().load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Loads the model inplace from a dumped `state_dict` object

        Parameters
        ----------
        destination: Any
        prefix: str
        keep_vars: bool

        Returns
        -------
        dict
        """
        state = super().state_dict(destination, prefix, keep_vars)
        state["cfg"] = self.cfg
        return state

    def set_n_labels(self, n_labels):
        """
        Sets the number of labels. To instantiate the linear layer, we need to
        call the `initialize` method.

        Parameters
        ----------
        n_labels: int
            Number of different labels predicted by this module
        """
        self.cfg["n_labels"] = n_labels

    def initialize(self):
        """
        Once the number of labels n_labels are known, this method
        initializes the torch linear layer.
        """
        raise NotImplementedError()

    def forward(
        self,
        embeds: torch.FloatTensor,
        mask: torch.BoolTensor,
        *,
        additional_outputs: typing.Dict[str, Any] = None,
        is_train: bool = False,
        is_predict: bool = False,
    ) -> Optional[torch.FloatTensor]:
        """
        Apply the nested pytorch module to:
        - compute the loss
        - predict the outputs
        non exclusively.
        If outputs are predicted, they are assigned to the `additional_outputs`
        list.

        Parameters
        ----------
        embeds: torch.FloatTensor
            Input embeddings
        mask: torch.BoolTensor
            Input embeddings mask
        additional_outputs: List
            Additional outputs that should not / cannot be back-propped through
            (Thinc treats Pytorch models solely as derivable functions, but the CRF
            that we employ performs the best tag decoding function with Pytorch)
            This list will contain the predicted outputs
        is_train: bool=False
            Are we training the model (defaults to True)
        is_predict: bool=False
            Are we predicting the model (defaults to False)

        Returns
        -------
        Optional[torch.FloatTensor]
            Optional 0d loss (shape = [1]) to train the model
        """
        raise NotImplementedError()


def custom_xp2torch(model, X):
    main = xp2torch(X[0], requires_grad=True)
    rest = convert_recursive(is_xp_array, lambda x: xp2torch(x), X[1:])

    def reverse_conversion(dXtorch):
        dX = torch2xp(dXtorch.args[0])
        return dX

    return (main, *rest), reverse_conversion


def pytorch_forward(
    model: Model,
    X: Tuple[Iterable[Doc], PredT, bool],
    is_train: bool = False,
) -> Tuple[Tuple[Floats1d, PredT], Callable[[Floats1d], Any]]:
    """
    Run the stacked CRF pytorch model to train / run a nested NER model

    Parameters
    ----------
    model: Model
    X: Tuple[Iterable[Doc], PredictionT, bool]
    is_train: bool

    Returns
    -------
    Tuple[Tuple[Floats1d, PredictionT], Callable[Floats1d, Any]]
    """
    [docs, *rest_X, is_predict] = X
    encoder: Model[List[Doc], List[Floats2d]] = model.get_ref("encoder")
    embeds_list, bp_embeds = encoder(docs, is_train=is_train)
    embeds = model.ops.pad(embeds_list)  # pad embeds

    ##################################################
    # Prepare the torch nested ner crf module inputs #
    ##################################################
    additional_outputs = {}
    # Convert input from numpy/cupy to torch
    (torch_embeds, *torch_rest), get_d_embeds = custom_xp2torch(
        model, (embeds, *rest_X)
    )
    # Prepare token mask from docs' lengths
    torch_mask = (
        torch.arange(embeds.shape[1], device=torch_embeds.device)
        < torch.tensor([d.shape[0] for d in embeds_list], device=torch_embeds.device)[
            :, None
        ]
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

    def backprop(d_loss: Floats1d) -> Any:
        d_loss_torch = ArgsKwargs(
            args=((loss_torch,),), kwargs={"grad_tensors": xp2torch(d_loss)}
        )
        d_embeds_torch = torch_backprop(d_loss_torch)
        d_embeds = get_d_embeds(d_embeds_torch)
        d_embeds_list = [
            d_padded_row[: len(d_item)]
            for d_item, d_padded_row in zip(embeds_list, d_embeds)
        ]
        d_docs = bp_embeds(d_embeds_list)
        return d_docs

    return (loss, additional_outputs), backprop


def instance_init(model: Model, X: List[Doc] = None, Y: Ints2d = None) -> Model:
    """
    Initializes the model by setting the input size of the model layers and the number
    of predicted labels

    Parameters
    ----------
    model: Model
        Nested NER thinc model
    X: List[Doc]
        list of documents on which we apply the encoder layer
    Y: Ints2d
        Unused gold spans

    Returns
    -------

    """
    encoder = model.get_ref("encoder")
    if X is not None:
        encoder.initialize(X)

    pt_model = model.attrs["pt_model"]
    pt_model.cfg["input_size"] = encoder.get_dim("nO")
    pt_model.initialize()
    pt_model.to(get_torch_default_device())
    model.set_dim("nI", pt_model.input_size)

    return model


@typing.overload
def wrap_pytorch_model(
    pt_model: PytorchWrapperModule,
    encoder: Model[List[Doc], List[Floats2d]],
) -> Model[
    Tuple[Iterable[Doc], Optional[PredT], Any, Any, Any, Optional[bool]],
    Tuple[Floats1d, PredT],
]:
    ...


@typing.overload
def wrap_pytorch_model(
    pt_model: PytorchWrapperModule,
    encoder: Model[List[Doc], List[Floats2d]],
) -> Model[
    Tuple[Iterable[Doc], Optional[PredT], Any, Any, Optional[bool]],
    Tuple[Floats1d, PredT],
]:
    ...


@typing.overload
def wrap_pytorch_model(
    pt_model: PytorchWrapperModule,
    encoder: Model[List[Doc], List[Floats2d]],
) -> Model[
    Tuple[Iterable[Doc], Optional[PredT], Any, Optional[bool]],
    Tuple[Floats1d, PredT],
]:
    ...


def wrap_pytorch_model(
    encoder: Model[List[Doc], List[Floats2d]],
    pt_model: PytorchWrapperModule,
) -> Model[
    Tuple[Iterable[Doc], Optional[PredT], Optional[bool]],
    Tuple[Floats1d, PredT],
]:
    """
    Chain and wraps a spaCy/Thinc encoder model (like a tok2vec) and a pytorch model.
    The loss should be computed directly in the Pytorch module and Categorical
    predictions are supported

    Parameters
    ----------
    encoder: Model[List[Doc], List[Floats2d]]
        The Thinc document token embedding layer
    pt_model: PytorchWrapperModule
        The Pytorch model

    Returns
    -------
        Tuple[Iterable[Doc], Optional[PredT], Optional[bool]],
        # inputs (docs, gold, *rest, is_predict)
        Tuple[Floats1d, PredT],
        # outputs (loss, *additional_outputs)
    """
    return Model(
        "pytorch",
        pytorch_forward,
        attrs={
            "set_n_labels": pt_model.set_n_labels,
            "pt_model": pt_model,
        },
        layers=[encoder],
        shims=[PyTorchShim(pt_model)],
        refs={"encoder": encoder},
        dims={"nI": None, "nO": None},
        init=instance_init,
    )
