from functools import partial
from typing import Any, Dict, List, Optional, Type, Union

import spacy
from pydantic import BaseModel, Extra, validator
from spacy.tokens import Doc, Span, Token

from edsnlp.utils.extensions import rsetattr

Assignable = Union[Type[Doc], Type[Span], Type[Token]]


def align_preds_and_spacy_data(preds, spacy_data):
    if isinstance(preds, dict):
        for i, single_spacy_data in enumerate(spacy_data):
            yield {k: v[i] for k, v in preds.items()}, single_spacy_data
    else:
        yield from zip(preds, spacy_data)


@spacy.registry.annotation_setters("set-all")
def configured_set_all():
    return set_all


def set_all(
    preds: Union[List[Dict], Dict[str, List[Any]]],
    spacy_data: List[Assignable],
):
    """
    Simple annotation setter.
    For a spaCy data (spacy_data[i]) and a prediction (preds[i]),
    will do in pseudo-code:
        spaCy_data._.k = v for k,v in prediction.items()

    Parameters
    ----------
    preds : Union[List[Dict], Dict[str, List[Any]]]
        Predictions, output of the model
    spacy_data : List[Assignable]
        List of Doc, Token or Span from which the predictions were made
    """
    for pred, single_spacy_data in align_preds_and_spacy_data(preds, spacy_data):
        for k, v in pred.items():
            rsetattr(single_spacy_data, f"_.{k}", v)


@spacy.registry.annotation_setters("from-mapping")
def configured_assign_from_dict(
    mapping,
):
    mapping = AssignModel.parse_obj(mapping).dict()["__root__"]
    return partial(
        assign_from_dict,
        mapping=mapping,
    )


def assign_from_dict(
    preds: Union[List[Dict], Dict[str, List[Any]]],
    spacy_data: List[Assignable],
    mapping: Dict[str, Dict[str, Assignable]],
):
    """
    Simple annotation setter.
    For a spaCy data (spacy_data[i]) and a prediction (preds[i]),
    will do in pseudo-code:
        spaCy_data.v = prediction[k]  k,v in mapping.items()

    Parameters
    ----------
    preds : Union[List[Dict], Dict[str, List[Any]]]
        Predictions, output of the model
    spacy_data : List[Assignable]
        List of Doc, Token or Span from which the predictions were made
    mapping : Dict[str, Dict[str, Assignable]]
        Dictionary were keys are keys of `preds`, and values are extensions to set
        on `spacy_data`
    """
    for pred, single_spacy_data in align_preds_and_spacy_data(preds, spacy_data):
        for k, extension_dict in mapping.items():
            rsetattr(single_spacy_data, extension_dict["extension"], pred[k])


class BaseAssignModel(BaseModel, arbitrary_types_allowed=True, extra=Extra.forbid):
    extension: str
    assigns: Optional[Assignable]

    @validator("assigns", pre=True)
    def assigns_validation(cls, v):
        if v is None:
            v = Span
        return v


class AssignModel(BaseModel, extra=Extra.forbid):

    __root__: Dict[str, Union[str, BaseAssignModel]]

    @validator("__root__", pre=True)
    def str_to_dict(cls, root):
        for k, v in root.items():
            if not isinstance(v, dict):
                root[k] = dict(extension=v, assigns=Span)
        return root
