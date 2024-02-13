import copy
import itertools
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Sequence,
    TypeVar,
    Union,
)

T = TypeVar("T")


def ld_to_dl(ld: Iterable[Mapping[str, T]]) -> Dict[str, List[T]]:
    """
    Convert a list of dictionaries to a dictionary of lists

    Parameters
    ----------
    ld: Iterable[Mapping[str, T]]
        The list of dictionaries

    Returns
    -------
    Dict[str, List[T]]
        The dictionary of lists
    """
    ld = list(ld)
    return {k: [dic.get(k) for dic in ld] for k in (ld[0] if len(ld) else ())}


def dl_to_ld(dl: Mapping[str, Sequence[Any]]) -> Iterator[Dict[str, Any]]:
    """
    Convert a dictionary of lists to a list of dictionaries

    Parameters
    ----------
    dl: Mapping[str, Sequence[Any]]
        The dictionary of lists

    Returns
    -------
    List[Dict[str, Any]]
        The list of dictionaries
    """
    return (dict(zip(dl, t)) for t in zip(*dl.values()))


FLATTEN_TEMPLATE = """\
def flatten(root):
    res={}
    return res
"""


def _discover_scheme(obj):
    keys = defaultdict(lambda: [])

    def rec(current, path):
        if not isinstance(current, dict):
            keys[id(current)].append(path)
            return
        for key, value in current.items():
            if not key.startswith("$"):
                rec(value, (*path, key))

    rec(obj, ())

    code = FLATTEN_TEMPLATE.format(
        "{"
        + "\n".join(
            "{}: root{},".format(
                repr("|".join(map("/".join, key_list))),
                "".join(f"[{repr(k)}]" for k in key_list[0]),
            )
            for key_list in keys.values()
        )
        + "}"
    )
    return code


class batch_compress_dict:
    """
    Compress a sequence of dictionaries in which values that occur multiple times are
    deduplicated. The corresponding keys will be merged into a single string using
    the "|" character as a separator.
    This is useful to preserve referential identities when decompressing the dictionary
    after it has been serialized and deserialized.

    Parameters
    ----------
    seq: Iterable[Dict[str, Any]]
        Sequence of dictionaries to compress
    """

    __slots__ = ("flatten", "seq")

    def __init__(self, seq: Iterable[Dict[str, Any]]):
        self.seq = seq
        self.flatten = None

    def __iter__(self):
        return batch_compress_dict(iter(self.seq))

    # def __getstate__(self):
    #     return {"seq": self.seq}

    # def __setstate__(self, state):
    #     self.seq = state["seq"]
    #     self.flatten = None

    def __next__(self) -> Dict[str, List]:
        exec_result = {}

        item = next(self.seq)
        if self.flatten is None:
            exec(_discover_scheme(item), {}, exec_result)
            self.flatten = exec_result["flatten"]
        return self.flatten(item)


def decompress_dict(seq: Union[Iterable[Dict[str, Any]], Dict[str, Any]]):
    """
    Decompress a dictionary of lists into a sequence of dictionaries.
    This function assumes that the dictionary structure was obtained using the
    `batch_compress_dict` class.
    Keys that were merged into a single string using the "|" character as a separator
    will be split into a nested dictionary structure.

    Parameters
    ----------
    seq: Union[Iterable[Dict[str, Any]], Dict[str, Any]]
        The dictionary to decompress or a sequence of dictionaries to decompress

    Returns
    -------

    """
    obj = ld_to_dl(seq) if isinstance(seq, Sequence) else seq
    res = {}
    for key, value in obj.items():
        for path in key.split("|"):
            current = res
            parts = path.split("/")
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
    return res


def batchify(
    iterable: Iterable[T],
    batch_size: int,
    drop_last: bool = False,
    formula: Callable = len,
) -> Iterable[List[T]]:
    """
    Yields batch that contain at most `batch_size` elements.
    If an item contains more than `batch_size` elements, it will be yielded as a single
    batch.

    Parameters
    ----------
    iterable
    batch_size
    """
    batch = []
    for item in iterable:
        next_size = formula(batch + [item])
        if next_size > batch_size and len(batch) > 0:
            yield batch
            batch = []
        batch.append(item)
    if len(batch) > 0 and not drop_last:
        yield batch


def get_attr_item(base, attr):
    try:
        return base[attr]
    except (KeyError, TypeError):
        return getattr(base, attr)


def split_names(names):
    _names = []
    for part in names.split("."):
        try:
            _names.append(int(part))
        except ValueError:
            _names.append(part)
    return _names


def get_deep_attr(base, names):
    if isinstance(names, str):
        names = split_names(names)
    if len(names) == 0:
        return base
    [current, *remaining] = names
    return get_deep_attr(get_attr_item(base, current), remaining)


def set_attr_item(base, attr, val):
    try:
        base[attr] = val
    except (KeyError, TypeError):
        setattr(base, attr, val)
    return base


def set_deep_attr(base, names, val):
    if isinstance(names, str):
        names = split_names(names)
    if len(names) == 0:
        return val
    if len(names) == 1:
        if isinstance(base, (dict, list)):
            base[names[0]] = val
        else:
            setattr(base, names[0], val)
    [current, *remaining] = names
    attr = base[current] if isinstance(base, (dict, list)) else getattr(base, current)
    try:
        set_deep_attr(attr, remaining, val)
    except TypeError:
        new_attr = list(attr)
        set_deep_attr(new_attr, remaining, val)
        return set_attr_item(base, current, tuple(new_attr))
    return base


class multi_tee:
    """
    Makes copies of an iterable such that every iteration over it
    starts from 0. If the iterable is a sequence (list, tuple), just returns
    it since every iter() over the object restart from the beginning
    """

    def __new__(cls, iterable):
        if isinstance(iterable, Sequence):
            return iterable
        return super().__new__(cls)

    def __init__(self, iterable):
        self.main, self.copy = itertools.tee(iterable)

    def __iter__(self):
        if self.copy is not None:
            it = self.copy
            self.copy = None
            return it
        return copy.copy(self.main)


class FrozenDict(dict):
    """
    Copied from `spacy.util.SimpleFrozenDict` to ensure compatibility.


    """

    def __init__(self, *args, error: str = None, **kwargs) -> None:
        """Initialize the frozen dict. Can be initialized with pre-defined
        values.

        error (str): The error message when user tries to assign to dict.
        """
        from spacy import Errors

        if error is None:
            error = Errors.E095
        super().__init__(*args, **kwargs)
        self.error = error

    def __setitem__(self, key, value):
        raise NotImplementedError(self.error)

    def pop(self, key, default=None):
        raise NotImplementedError(self.error)

    def update(self, other):
        raise NotImplementedError(self.error)


class FrozenList(list):
    """
    Copied from `spacy.util.SimpleFrozenDict` to ensure compatibility
    """

    def __init__(self, *args, error: str = None) -> None:
        """Initialize the frozen list.

        error (str): The error message when user tries to mutate the list.
        """
        from spacy import Errors

        if error is None:
            error = Errors.E927
        self.error = error
        super().__init__(*args)

    def append(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def clear(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def extend(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def insert(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def pop(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def remove(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def reverse(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def sort(self, *args, **kwargs):
        raise NotImplementedError(self.error)


def flatten_once(items):
    for item in items:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def flatten(items):
    for item in items:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item
