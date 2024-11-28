from __future__ import annotations

import random
from typing import Any, Callable, Iterable, Optional, Union

import pandas as pd
from typing_extensions import Literal

from edsnlp import registry
from edsnlp.core.stream import Stream
from edsnlp.data.base import BaseWriter, MemoryBasedReader
from edsnlp.data.converters import get_dict2doc_converter, get_doc2dict_converter
from edsnlp.utils.collections import dl_to_ld, flatten, ld_to_dl
from edsnlp.utils.stream_sentinels import DatasetEndSentinel
from edsnlp.utils.typing import AsList


class PandasReader(MemoryBasedReader):
    DATA_FIELDS = ("data",)

    def __init__(
        self,
        data: pd.DataFrame,
        shuffle: Literal["dataset", False] = False,
        seed: Optional[int] = None,
        loop: bool = False,
    ):
        super().__init__()
        self.shuffle = shuffle
        seed = seed if seed is not None else random.getrandbits(32)
        self.rng = random.Random(seed)
        self.emitted_sentinels = {"dataset"}
        self.loop = loop
        self.data = data
        assert isinstance(data, pd.DataFrame)

    def read_records(self) -> Iterable[Any]:
        while True:
            data = self.data
            if self.shuffle == "dataset":
                data = data.sample(frac=1.0, random_state=self.rng.getrandbits(32))
            yield from dl_to_ld(dict(data))
            yield DatasetEndSentinel()
            if not self.loop:
                break

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={object.__repr__(self.data)}, "
            f"shuffle={self.shuffle}, "
            f"loop={self.loop})"
        )


@registry.readers.register("pandas")
def from_pandas(
    data,
    converter: Optional[AsList[Union[str, Callable]]] = None,
    shuffle: Literal["dataset", False] = False,
    seed: Optional[int] = None,
    loop: bool = False,
    **kwargs,
) -> Stream:
    """
    The PandasReader (or `edsnlp.data.from_pandas`) handles reading from a table and
    yields documents. At the moment, only entities and attributes are loaded. Relations
    and events are not supported.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)
    doc_iterator = edsnlp.data.from_pandas(df, nlp=nlp, converter="omop")
    annotated_docs = nlp.pipe(doc_iterator)
    ```

    !!! note "Generator vs list"

        `edsnlp.data.from_pandas` returns a
        [Stream][edsnlp.core.stream.Stream].
        To iterate over the documents multiple times efficiently or to access them by
        index, you must convert it to a list

        ```{ .python .no-check }
        docs = list(edsnlp.data.from_pandas(df, converter="omop"))
        ```

    Parameters
    ----------
    data: pd.DataFrame
        Pandas object
    shuffle: Literal["dataset", False]
        Whether to shuffle the data. If "dataset", the whole dataset will be shuffled
        before starting iterating on it (at the start of every epoch if looping).
    seed: Optional[int]
        The seed to use for shuffling.
    loop: bool
        Whether to loop over the data indefinitely.
    converter: Optional[AsList[Union[str, Callable]]]
        Converters to use to convert the rows of the DataFrame (represented as dicts)
        to Doc objects. These are documented on the [Converters](/data/converters) page.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.

    Returns
    -------
    Stream
    """

    data = Stream(
        reader=PandasReader(
            data,
            shuffle=shuffle,
            seed=seed,
            loop=loop,
        )
    )
    if converter:
        for conv in converter:
            conv, kwargs = get_dict2doc_converter(conv, kwargs)
            data = data.map(conv, kwargs=kwargs)
    return data


class PandasWriter(BaseWriter):
    def __init__(self, dtypes: Optional[dict] = None):
        self.dtypes = dtypes

    def consolidate(self, items):
        columns = ld_to_dl(flatten(items))
        res = pd.DataFrame(columns)
        return res.astype(self.dtypes) if self.dtypes else res


@registry.writers.register("pandas")
def to_pandas(
    data: Union[Any, Stream],
    execute: bool = True,
    converter: Optional[Union[str, Callable]] = None,
    dtypes: Optional[dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    `edsnlp.data.to_pandas` writes a list of documents as a pandas table.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)

    doc = nlp("My document with entities")

    edsnlp.data.to_pandas([doc], converter="omop")
    ```

    Parameters
    ----------
    data: Union[Any, Stream],
        The data to write (either a list of documents or a Stream).
    dtypes: Optional[dict]
        Dictionary of column names to dtypes. This is passed to `pd.DataFrame.astype`.
    execute: bool
        Whether to execute the writing operation immediately or to return a stream
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the documents to dictionary objects before storing
        them in the dataframe. These are documented on the
        [Converters](/data/converters) page.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented on
        the [Converters](/data/converters) page.
    """
    data = Stream.ensure_stream(data)
    if converter:
        converter, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)

    return data.write(PandasWriter(dtypes), execute=execute)
