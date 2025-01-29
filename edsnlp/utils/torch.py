import copyreg
import math
import warnings
from enum import Enum
from typing import TypeVar

import dill
import torch

# filter "is in beta" torch warnings
warnings.filterwarnings("ignore", message=".*is in beta.*")

Args = TypeVar("Args")


def pad_2d(data, pad=0, **kwargs):
    max_len = max(map(len, data), default=0)
    padded = [row + [pad] * (max_len - len(row)) for row in data]
    return torch.as_tensor(padded, **kwargs)


class ActivationFunction(str, Enum):
    relu = "relu"
    gelu = "gelu"
    glu = "glu"


def get_activation_function(activation: ActivationFunction):
    return getattr(torch.nn.functional, activation)


def mask_to_triangle(mask):
    """
    Convert a boolean mask to a tensor containing distance to the nearest
    edge of the mask, in each direction.
    For example, if the mask is
    ```
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
    ]
    ```

    The output will be
    ```
    [
        [0, 1, 2, 1, 0],
        [0, 1, 0, -1, -2]
    ]
    ```

    Parameters
    ----------
    mask: torch.Tensor
        A boolean mask

    Returns
    -------
    torch.Tensor
    """
    ramp = torch.arange(0, mask.shape[1], 1)
    scores = torch.min(ramp, mask.sum(1, keepdim=True) - 1 - ramp).view(-1)
    return scores


def make_windows(lengths, size, stride):
    max_len = max(lengths)
    windows = pad_2d(
        [
            list(
                range(
                    idx * stride + doc_idx * max_len,
                    min(idx * stride + size, length) + doc_idx * max_len,
                )
            )
            for doc_idx, length in enumerate(lengths)
            for idx in range(0, 1 + max(0, math.ceil((length - size) / stride)))
        ],
        pad=-1,
        dtype=torch.long,
    )
    windows_mask = windows != -1
    windows[~windows_mask] = 0
    indexer = torch.zeros((len(lengths), max_len), dtype=torch.long).view(-1)
    scores = mask_to_triangle(windows_mask)
    scores = scores * len(scores) + torch.arange(len(scores))
    scores[~windows_mask.view(-1)] = -1
    indexer.index_reduce_(
        dim=0,
        source=scores,
        index=windows.view(-1),
        reduce="amax",
    )
    indexer %= len(scores)
    return windows.masked_fill(~windows_mask, -1), indexer


def reduce_empty(*args, **kwargs):
    return type(None), ()


def load_pruned_obj(obj, _):
    return obj


# Torch may still be imported as a namespace package, so we can access the
# torch.save and torch.load functions

MAP_LOCATION = None


try:
    from accelerate.hooks import AlignDevicesHook

    # We need to replace the "execution_device" attribute of the AlignDevicesHook
    # using map_location when unpickling the stream

    def save_align_devices_hook(pickler, obj):
        pickler.save_reduce(load_align_devices_hook, (obj.__dict__,), obj=obj)

    def load_align_devices_hook(state):
        state["execution_device"] = MAP_LOCATION
        new_obj = AlignDevicesHook.__new__(AlignDevicesHook)
        new_obj.__dict__.update(state)
        return new_obj

except ImportError:
    AlignDevicesHook = None


def dump(
    *args,
    skip_tensors: bool = False,
    **kwargs,
):
    # We need to replace the "execution_device" attribute of the AlignDevicesHook
    # using map_location when pickling the stream
    old = None
    old_settings = dict(dill.settings)
    old_dispatch = {}
    try:
        if skip_tensors:
            if torch.Tensor in copyreg.dispatch_table:
                old_dispatch[torch.Tensor] = copyreg.dispatch_table[torch.Tensor]
            copyreg.pickle(torch.Tensor, reduce_empty)
        if AlignDevicesHook is not None:
            old = dill.Pickler.dispatch.get(AlignDevicesHook)
            dill.Pickler.dispatch[AlignDevicesHook] = save_align_devices_hook
        dill.settings["recurse"] = False
        dill.settings["byref"] = True
        return torch.save(*args, pickle_module=dill, **kwargs)
    finally:
        dill.settings.update(old_settings)
        if AlignDevicesHook is not None:
            del dill.Pickler.dispatch[AlignDevicesHook]
            if old is not None:  # pragma: no cover
                dill.Pickler.dispatch[AlignDevicesHook] = old
        copyreg.dispatch_table.pop(torch.Tensor, None)
        copyreg.dispatch_table.update(old_dispatch)


def load(*args, map_location=None, **kwargs):
    global MAP_LOCATION
    MAP_LOCATION = map_location
    if torch.__version__ >= "2.1" and isinstance(args[0], str):
        kwargs["mmap"] = True
    try:
        if torch.__version__ < "2.0.0":
            torch.load.__globals__["pickle"] = dill
        result = torch.load(
            *args,
            pickle_module=dill,
            map_location=map_location,
            **kwargs,
        )
    finally:
        import pickle

        torch.load.__globals__["pickle"] = pickle
    MAP_LOCATION = None
    return result
