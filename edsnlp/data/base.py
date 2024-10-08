from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

from edsnlp.core.stream import Stream

from ..utils.collections import flatten
from .converters import FILENAME, get_dict2doc_converter, get_doc2dict_converter


class BaseReader:
    """
    The BaseReader servers as a base class for all readers. It expects two methods:

    - `read_records` method which is called in the main process and should return a
        generator of fragments (like filenames) with their estimated size (number of
        documents)
    - `unpack_tasks` method which is called in the worker processes and receives
        batches of fragments and should return a list of dictionaries (one per
        document), ready to be converted to a Doc object by the converter.

    Additionally, the subclass should define a `DATA_FIELDS` class attribute which
    contains the names of all attributes that should not be copied when the reader is
    copied to the worker processes. This is useful for example when the reader holds a
    reference to a large object like a DataFrame that should not be copied to the
    worker processes.
    """

    DATA_FIELDS = ()
    read_in_worker: bool

    def read_records(self) -> Iterable[Any]:
        raise NotImplementedError()

    def worker_copy(self):
        if self.read_in_worker:
            return self
        # new reader without data, this will not call __init__ since we use __dict__
        # to set the data
        reader = self.__class__.__new__(self.__class__)
        state = {
            k: v
            for k, v in self.__dict__.items()
            if k not in self.__class__.DATA_FIELDS
        }
        reader.__dict__ = state
        return reader


class FileBasedReader(BaseReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_in_worker = True


class MemoryBasedReader(BaseReader):
    def __init__(self, read_in_worker: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_in_worker = False

    def __repr__(self):
        return f"{self.__class__.__name__}(data={object.__repr__(self.data)})"


class BaseWriter:
    def handle_record(self, record: Union[Dict, List[Dict]]):
        for subitem in flatten(record):
            if isinstance(subitem, dict):
                subitem.pop(FILENAME, None)
        return record

    def consolidate(self, items: Iterable):
        raise NotImplementedError()


class BatchWriter(BaseWriter):
    batch_size: Optional[int] = None
    batch_by: Callable
    batch_in_worker: bool = False

    def handle_batch(self, batch):
        raise NotImplementedError()


class IterableReader(MemoryBasedReader):
    DATA_FIELDS = ("data",)

    def __init__(self, data: Iterable, read_in_worker: bool = False):
        self.data = data
        self.read_in_worker = read_in_worker

        super().__init__()

    def read_records(self) -> Iterable[Any]:
        return self.data


def from_iterable(
    data: Iterable,
    converter: Union[str, Callable] = None,
    read_in_worker: bool = False,
    **kwargs,
) -> Stream:
    """
    The IterableReader (or `edsnlp.data.from_iterable`) reads a list of Python objects (
    texts, dictionaries, ...) and yields documents by passing them through the
    `converter` if given, or returns them as is.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)
    doc_iterator = edsnlp.data.from_iterable([{...}], nlp=nlp, converter=...)
    annotated_docs = nlp.pipe(doc_iterator)
    ```

    !!! note "Generator vs list"

        `edsnlp.data.from_iterable` returns a
        [Stream][edsnlp.core.stream.Stream].
        To iterate over the documents multiple times efficiently or to access them by
        index, you must convert it to a list

        ```{ .python .no-check }
        docs = list(edsnlp.data.from_iterable([{...}], converter=...)
        ```

    Parameters
    ----------
    data: Iterable
        The data to read
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the JSON rows of the data source to Doc objects
    read_in_worker: bool
        In multiprocessing mode, whether to read the data in the worker processes.
        If `True`, the data will be read in the worker processes, requires pickling the
        input iterable: this is mostly useful if the pickled iterable is smaller than
        the data itself (eg, an infinite generator of synthetic data). If `False`, the
        data will be read in the main process and distributed to the workers.
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented
        on the [Converters](/data/converters) page.

    Returns
    -------
    Stream
    """
    if not isinstance(data, Stream):
        data = Stream(IterableReader(data, read_in_worker=read_in_worker))
    if converter:
        converter, kwargs = get_dict2doc_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)
    return data


def to_iterable(
    data: Union[Any, Stream],
    converter: Optional[Union[str, Callable]] = None,
    **kwargs,
):
    """
    `edsnlp.data.to_items` returns an iterator of documents, as converted by the
    `converter`. In comparison to just iterating over a Stream, this will
    also apply the `converter` to the documents, which can lower the data transfer
    overhead when using multiprocessing.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)

    doc = nlp("My document with entities")

    edsnlp.data.to_items([doc], converter="omop")
    ```

    Parameters
    ----------
    data: Union[Any, Stream],
        The data to write (either a list of documents or a Stream).
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the documents to dictionary objects.
    kwargs:
        Additional keyword arguments passed to the converter. These are documented
        on the [Converters](/data/converters) page.
    """
    data = Stream.ensure_stream(data)
    if converter:
        converter, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)

    return data
