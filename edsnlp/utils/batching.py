import warnings
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import Literal

from .stream_sentinels import (
    DATASET_END_SENTINEL,
    DatasetEndSentinel,
    FragmentEndSentinel,
    StreamSentinel,
)
from .typing import Validated

T = TypeVar("T")


class BatchSizeArg(Validated):
    """
    Batch size argument validator / caster for confit/pydantic

    Examples
    --------

    ```{ .python .no-check }
    def fn(batch_size: BatchSizeArg):
        return batch_size


    print(fn("10 samples"))
    # Out: (10, "samples")

    print(fn("10 words"))
    # Out: (10, "words")

    print(fn(10))
    # Out: (10, "samples")
    ```
    """

    @classmethod
    def validate(cls, value, config=None):
        value = str(value)
        parts = value.split()
        try:
            num = int(parts[0])
        except ValueError:
            num = None
        if len(parts) == 1 and num is not None:
            return num, "docs"
        elif len(parts) == 1 and value.isidentifier():
            return None, value
        if len(parts) == 2 and num is not None and parts[1].isidentifier():
            return num, parts[1]
        raise ValueError(f'Invalid batch size: {value}, must be <int> or "<int> <str>"')


if TYPE_CHECKING:
    BatchSizeArg = Union[str, Tuple[int, str]]  # noqa: F811


def batchify(
    iterable: Iterable[T],
    batch_size: int,
    drop_last: bool = False,
    sentinel_mode: Literal["drop", "keep", "split"] = "drop",
) -> Iterable[List[T]]:
    """
    Yields batch that contain at most `batch_size` elements.
    If an item contains more than `batch_size` elements, it will be yielded as a single
    batch.

    Parameters
    ----------
    iterable: Iterable[T]
        The iterable to batchify
    batch_size: int
        The maximum number of elements in a batch
    drop_last: bool
        Whether to drop the last batch if it is smaller than `batch_size`
    sentinel_mode: Literal["auto", "drop", "keep", "split"]
        How to handle the sentinel values in the iterable:

        - "drop": drop sentinel values
        - "keep": keep sentinel values inside the produced batches
        - "split": split batches at sentinel values and yield sentinel values
            separately
    """
    assert sentinel_mode in ("drop", "keep", "split")
    batch = []
    num_items = 0
    for item in iterable:
        if isinstance(item, StreamSentinel):
            if sentinel_mode == "split":
                if num_items > 0:
                    yield batch
                yield item
                batch = []
                num_items = 0
            elif sentinel_mode == "keep":
                batch.append(item)
            continue
        if num_items >= batch_size:
            yield batch
            batch = []
            num_items = 0
        batch.append(item)
        num_items += 1
    if num_items > 0 and not drop_last:
        yield batch


def batchify_by_length_sum(
    iterable: Iterable[T],
    batch_size: int,
    drop_last: bool = False,
    sentinel_mode: Literal["drop", "keep", "split"] = "drop",
) -> Iterable[List[T]]:
    """
    Yields batch that contain at most `batch_size` words.
    If an item contains more than `batch_size` words, it will be yielded as a single
    batch.

    Parameters
    ----------
    iterable: Iterable[T]
        The iterable to batchify
    batch_size: int
        The maximum number of words in a batch
    drop_last: bool
        Whether to drop the last batch if it is smaller than `batch_size`
    sentinel_mode: Literal["auto", "drop", "keep", "split"]
        How to handle the sentinel values in the iterable:

        - "drop": drop sentinel values
        - "keep": keep sentinel values inside the produced batches
        - "split": split batches at sentinel values and yield sentinel values
            separately

    Returns
    -------
    Iterable[List[T]]
    """
    assert sentinel_mode in ("drop", "keep", "split")
    batch = []
    total = 0
    num_items = 0
    for item in iterable:
        if isinstance(item, StreamSentinel):
            if sentinel_mode == "split":
                if len(batch) > 0:
                    yield batch
                yield item
                batch = []
                total = 0
                num_items = 0
            elif sentinel_mode == "keep":
                batch.append(item)
            continue
        count = len(item)
        if total + count > batch_size and num_items > 0:
            yield batch
            batch = []
            total = 0
            num_items = 0
        batch.append(item)
        num_items += 1
        total += count
    if num_items > 0 and not drop_last:
        yield batch


def batchify_by_padded(
    iterable: Iterable[T],
    batch_size: int,
    drop_last: bool = False,
    sentinel_mode: Literal["drop", "keep", "split"] = "drop",
) -> Iterable[List[T]]:
    """
    Yields batch that contain at most `batch_size` padded words, ie the number
    of total words if all items were padded to the length of the longest item.

    Parameters
    ----------
    iterable: Iterable[T]
        The iterable to batchify
    batch_size: int
        The maximum number of padded words in a batch
    drop_last: bool
        Whether to drop the last batch if it is smaller than `batch_size`
    sentinel_mode: Literal["auto", "drop", "keep", "split"]
        How to handle the sentinel values in the iterable:

        - "drop": drop sentinel values
        - "keep": keep sentinel values inside the produced batches
        - "split": split batches at sentinel values and yield sentinel values
            separately

    Returns
    -------
    Iterable[List[T]]
    """
    assert sentinel_mode in ("drop", "keep", "split")
    batch = []
    num_items = 0
    max_words = 0
    for item in iterable:
        if isinstance(item, StreamSentinel):
            if sentinel_mode == "split":
                if len(batch) > 0:
                    yield batch
                yield item
                batch = []
                num_items = 0
                max_words = 0
            elif sentinel_mode == "keep":
                batch.append(item)
            continue
        count = len(item)
        next_count = max(max_words, count)
        if (1 + num_items) * next_count > batch_size and num_items > 0:
            yield batch
            batch = []
            num_items = 0
            next_count = count
        batch.append(item)
        num_items += 1
        max_words = next_count
    if num_items > 0 and not drop_last:
        yield batch


def batchify_by_dataset(
    iterable: Iterable[T],
    batch_size: Optional[int] = None,
    drop_last: bool = False,
    sentinel_mode: Literal["drop", "keep", "split"] = "drop",
) -> Iterable[List[T]]:
    """
    Yields batch that contain at most `batch_size` datasets.
    If an item contains more than `batch_size` datasets, it will be yielded as a single
    batch.

    Parameters
    ----------
    iterable: Iterable[T]
        The iterable to batchify
    batch_size: Optional[int]
        Unused, always 1 full dataset per batch
    drop_last: bool
        Whether to drop the last batch if it is smaller than `batch_size`
    sentinel_mode: Literal["auto", "drop", "keep", "split"]
        How to handle the sentinel values in the iterable:

        - "drop": drop sentinel values
        - "keep": keep sentinel values inside the produced batches
        - "split": split batches at sentinel values and yield sentinel values
            separately

    Returns
    -------
    Iterable[List[T]]
    """
    assert sentinel_mode in ("drop", "keep", "split")
    assert batch_size is None, f"batch_size should be None, got {batch_size}"
    batch = []
    num_items = 0
    for item in iterable:
        is_end_dataset = item is DATASET_END_SENTINEL
        if isinstance(item, StreamSentinel):
            if sentinel_mode == "split" or is_end_dataset:
                if num_items > 0 or is_end_dataset:
                    yield batch
                yield item
                batch = []
                num_items = 0
                continue
            elif sentinel_mode == "keep":
                batch.append(item)
                continue
            else:  # drop
                continue
        batch.append(item)
        num_items += 1
    if num_items > 0 and not drop_last:
        yield batch


batchify_by_dataset.requires_sentinel = "dataset"


def batchify_by_fragment(
    iterable: Iterable[T],
    batch_size: Optional[int] = None,
    drop_last: bool = False,
    sentinel_mode: Literal["drop", "keep", "split"] = "drop",
) -> Iterable[List[T]]:
    """
    Yields batch that contain at most `batch_size` fragments.
    If an item contains more than `batch_size` fragments, it will be yielded as a single
    batch.

    Parameters
    ----------
    iterable: Iterable[T]
        The iterable to batchify
    batch_size: Optional[int]
        Unused, always 1 full fragment per batch
    drop_last: bool
        Whether to drop the last batch if it is smaller than `batch_size`
    sentinel_mode: Literal["auto", "drop", "keep", "split"]
        How to handle the sentinel values in the iterable:

        - "drop": drop sentinel values
        - "keep": keep sentinel values inside the produced batches
        - "split": split batches at sentinel values and yield sentinel values
            separately

    Returns
    -------
    Iterable[List[T]]
    """
    assert sentinel_mode in ("drop", "keep", "split")
    assert batch_size is None
    batch = []
    num_items = 0
    for item in iterable:
        is_end_fragment = isinstance(item, (FragmentEndSentinel, DatasetEndSentinel))
        if isinstance(item, StreamSentinel):
            if sentinel_mode == "split" or is_end_fragment:
                if num_items > 0 or is_end_fragment:
                    yield batch
                yield item
                batch = []
                num_items = 0
                continue
            elif sentinel_mode == "keep":
                batch.append(item)
                continue
            else:  # drop
                continue
        batch.append(item)
        num_items += 1
    if num_items > 0 and not drop_last:
        yield batch


batchify_by_fragment.requires_sentinel = "fragment"

BatchFn = Union[
    Callable[[Iterable, int, Literal["drop", "split"]], Iterable],
    Callable[[Iterable, int], Iterable],
]
BatchBy = Union[str, BatchFn]

batchify_fns = {
    "words": batchify_by_length_sum,
    "padded_words": batchify_by_padded,
    "dataset": batchify_by_dataset,
    "fragment": batchify_by_fragment,
    "docs": batchify,
}


def stat_batchify(key):
    """
    Create a batching function that uses the value of a specific key in the items to
    determine the batch size. This function is primarily meant to be used on the
    flattened outputs of the `preprocess` method of a
    [Pipeline][edsnlp.core.pipeline.Pipeline] object.

    It expects the items to be a dictionary in which some keys contain the string
    "/stats/" and the `key` pattern. For instance:

    ```python
    from edsnlp.utils.batching import stat_batchify

    items = [
        {"text": "first sample", "obj/stats/words": 2, "obj/stats/chars": 12},
        {"text": "dos", "obj/stats/words": 1, "obj/stats/chars": 3},
        {"text": "third one !", "obj/stats/words": 3, "obj/stats/chars": 11},
    ]
    batcher = stat_batchify("words")
    assert list(batcher(items, 4)) == [
        [
            {"text": "first sample", "obj/stats/words": 2, "obj/stats/chars": 12},
            {"text": "dos", "obj/stats/words": 1, "obj/stats/chars": 3},
        ],
        [
            {"text": "third one !", "obj/stats/words": 3, "obj/stats/chars": 11},
        ],
    ]
    ```


    Parameters
    ----------
    key: str
        The key pattern to use to determine the actual key to look up in the items.

    Returns
    -------
    Callable[[Iterable, int, bool, Literal["drop", "split"]], Iterable
    """

    def rec(
        iterable,
        batch_size,
        drop_last=False,
        sentinel_mode="drop",
    ):
        batch = []
        num_items = 0
        total = 0
        exact_key = None
        for item in iterable:
            if isinstance(item, StreamSentinel):
                if sentinel_mode == "split":
                    if num_items > 0:
                        yield batch
                    yield item
                    batch = []
                    num_items = 0
                    total = 0
                elif sentinel_mode == "keep":
                    batch.append(item)
                continue
            if exact_key is None:
                candidates = [k for k in item if "/stats/" in k and key in k]
                if len(candidates) != 1:
                    warnings.warn(
                        f"Batching key {key!r} should match one "
                        f"candidate in {[k for k in item if '/stats/' in k]}"
                    )
                if len(candidates) == 0:
                    stat_keys = [k for k in item if "/stats/"]
                    raise ValueError(
                        f"Pattern {key!r} doesn't match any key in {stat_keys} "
                        " to determine the batch size."
                    )
                exact_key = candidates[0]
            value = item[exact_key]
            if num_items > 0 and total + value > batch_size:
                yield batch
                batch = []
                num_items = 0
                total = 0
            total += value
            batch.append(item)
            num_items += 1
        if num_items > 0 and not drop_last:
            yield batch

    if key == "docs":
        return batchify

    return rec
