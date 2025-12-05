import ast
import random
import re
import warnings
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Union

from confit import validate_arguments

from edsnlp import registry
from edsnlp.core.stream import Stream
from edsnlp.data.base import FileBasedReader
from edsnlp.data.converters import get_dict2doc_converter, get_doc2dict_converter
from edsnlp.utils.stream_sentinels import DatasetEndSentinel


class HFDatasetReader(FileBasedReader):
    def __init__(
        self,
        dataset: Any,
        shuffle: Union[Literal["dataset"], bool] = False,
        seed: Optional[int] = None,
        loop: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.shuffle = "dataset" if shuffle == "dataset" or shuffle is True else False
        seed = seed if seed is not None else random.getrandbits(32) 
        self.rng = random.Random(seed)
        self.loop = loop
        self.emitted_sentinels = {"dataset"}

    def read_records(self) -> Iterable[Any]:
        while True:
            data = self.dataset
            if self.shuffle == "dataset":
                try:
                    data = list(self.dataset)
                except Exception:
                    # fallback to non-shuffled iterator
                    data = self.dataset
                else:
                    self.rng.shuffle(data)

            for item in data:
                yield item
       
            yield DatasetEndSentinel()
            if not self.loop:
                break

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={object.__repr__(self.dataset)}, "
            f"shuffle={self.shuffle}, "
            f"loop={self.loop})"
        )


def _import_datasets():
    try:
        import datasets
        return datasets
    except Exception as e:  # pragma: no cover - dependency handling
        raise ImportError(
            "The 'datasets' library is required to write huggingface datasets. "
            "Install it with `pip install datasets` or with the edsnlp extras: "
            "`pip install 'edsnlp[ml]'`"
        ) from e

def _validate_hf_ner_columns(col_names: Optional[Iterable[str]], kwargs: Dict[str, Any]):
        words_col = kwargs.get("words_column", "tokens")
        ner_col = kwargs.get("ner_tags_column", "ner_tags")

        missing = []
        if words_col not in col_names:
            missing.append(f"words_column=\"{words_col}\"")
        if ner_col not in col_names:
            missing.append(f"ner_tags_column=\"{ner_col}\"")
        if missing:
            raise ValueError(
                "Cannot find these columns in dataset: "
                f"{missing}. "
                + f"Dataset columns are: {col_names}."
                + (
                    " If you intended to process raw text, consider using the "
                    "'hf_text' converter (pass `converter='hf_text'` and "
                    "`text_column='<column>'`)."
                )
            )

        kwargs["words_column"] = words_col
        kwargs["ner_tags_column"] = ner_col
        
        if "id" in col_names and "id_column" not in kwargs:
            kwargs["id_column"] = "id"

def _validate_hf_text_columns(col_names: Optional[Iterable[str]], kwargs: Dict[str, Any]):
        text_col = kwargs.get("text_column", "text")

        missing = []
        if text_col not in col_names:
                missing.append(f"text_column=\"{text_col}\"")
        if missing:
            raise ValueError(
                "Cannot find these columns in dataset: "
                f"{missing}. "
                + f"Dataset columns are: {col_names}."
                + (
                    " If you intended to process a NER dataset, consider using the "
                    "'hf_ner' converter (pass `converter='hf_ner')."
                )
            )

        kwargs["text_column"] = text_col
        
        if "id" in col_names and "id_column" not in kwargs:
            kwargs["id_column"] = "id"

def _load_hf_dataset_with_config(
    dataset: str,
    split: Optional[str] = None,
    name: Optional[str] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    datasets = _import_datasets()

    try:
        ds = datasets.load_dataset(dataset, name=name, split=split, **load_kwargs)
    except ValueError as e:
        msg = str(e)
        # Handle datasets that require a config name when none was provided
        if "Config name is missing" not in msg or name is not None:
            raise
            
        m = re.search(r"available configs: (\[.*\])", msg)
        if not m:
            raise ValueError(
                f"Config name is missing for dataset {dataset!r}. "
                f"Please pass a `name` among the available configs as "
                f"reported by the dataset builder."
            ) from e
            
        try:
            configs = ast.literal_eval(m.group(1))
            if not configs:
                raise
            chosen = configs[0]
            warnings.warn(
                f"Dataset {dataset!r} requires a config name; "
                f"no `name` was provided. Using first available config "
                f"'{chosen}'. Pass `name` to select another config among: {configs}.",
                UserWarning,
            )
            ds = datasets.load_dataset(
                dataset, name=chosen, split=split, **load_kwargs
            )
        except Exception:
            raise ValueError(
                f"Config name is missing for dataset {dataset!r}. "
                f"Please pass a `name` among the available configs as "
                f"reported by the dataset builder."
            ) from e
    return ds

@registry.readers.register("huggingface_dataset ")
@validate_arguments()
def from_huggingface_dataset (
    dataset: Union[str, Any],
    split: Optional[str] = None,
    name: Optional[str] = None,  # Add config/subset name parameter
    shuffle: Union[Literal["dataset"], bool] = False,
    seed: Optional[int] = None,
    loop: bool = False,
    converter: Optional[Union[str, Callable]] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Stream:
    """
    Load a dataset from the HuggingFace Hub as a Stream.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)
    doc_iterator = edsnlp.data.from_huggingface_dataset (
        "lhoestq/conll2003",
        split="train",
        tag_order=[
            'O',
            'B-PER',
            'I-PER',
            'B-ORG',
            'I-ORG',
            'B-LOC',
            'I-LOC',
            'B-MISC',
            'I-MISC',
        ],
        converter="hf_ner",
    )
    annotated_docs = nlp.pipe(doc_iterator)
    ```

    Parameters
    ----------
    dataset: Union[str, Any]
        Either a dataset identifier (e.g. "conll2003") or an already loaded
        `datasets.Dataset` / `datasets.IterableDataset` object.
    split: Optional[str]
        Which split to load (e.g. "train"). If None, the default dataset split
        returned by `datasets.load_dataset` is used.
    name: Optional[str]
        Configuration name for datasets with multiple configs (e.g. "en" for
        a multilingual dataset). Also known as the subset name.
    converter: Optional[Union[str, Callable]]
        Converter(s) to transform dataset dicts to Doc objects. Recommended
        converters are `"hf_ner"` and `"hf_text"`. More information is available
        in the [Converters](/data/converters) page.
    shuffle: Literal['dataset', False] or bool
        Whether to shuffle the dataset before yielding. If True or 'dataset', the
        whole dataset will be materialized and shuffled (may be expensive).
    seed: Optional[int]
        Random seed for shuffling.
    loop: bool
        Whether to loop over the dataset indefinitely.
    load_kwargs: dict
        Dictionary of additional kwargs that will be passed to the
        `datasets.load_dataset()` method.
    kwargs: dict
        Additional keyword arguments passed to the converter, these are
        documented in the [Converters](/data/converters) page.

    Returns
    -------
    Stream
    """
    load_kwargs = load_kwargs or {}

    # If user passed a dataset identifier string, load it
    ds = dataset
    if isinstance(dataset, str):
        ds = _load_hf_dataset_with_config(
            dataset, split=split, name=name, load_kwargs=load_kwargs
        )

    else:
        # If user passed a (split) name to select
        if split is not None:
            try:
                ds = dataset[split]
            except Exception:
                raise ValueError(
                    f"Cannot select split {split!r} from dataset {dataset!r}."
                )
    
    # If no split was provided and the loaded dataset exposes multiple splits
    # (e.g., a `DatasetDict`), pick the 'train' split by default and warn
    # the user to be explicit.
    if split is None and hasattr(ds, "keys") and "train" in ds.keys():
        warnings.warn(
            f"Dataset {dataset!r} contains multiple splits and no `split` "
            f"was provided; using 'train' by default. Pass `split` to "
            f"select another split.",
            UserWarning,
        )
        ds = ds["train"]

    if "hf_ner" in converter:
        _validate_hf_ner_columns(list(ds.column_names), kwargs)
    if "hf_text" in converter:
        _validate_hf_text_columns(list(ds.column_names), kwargs)
        
    reader = HFDatasetReader(ds, shuffle=shuffle, seed=seed, loop=loop)
    stream = Stream(reader=reader)

    if converter:
        conv, kwargs = get_dict2doc_converter(converter, kwargs)
        stream = stream.map(conv, kwargs=kwargs)

    return stream



def _iter_from_stream(data_stream):
    for item in data_stream.execute():
        if isinstance(item, DatasetEndSentinel):
            continue
        if isinstance(item, (list, tuple)):
            for rec in item:
                yield rec
        else:
            yield item


@registry.writers.register("huggingface_dataset ")
def to_huggingface_dataset (
    data: Union[Any, Stream],
    *,
    converter: Optional[Union[str, Callable]] = None,
    execute: bool = True,
    **kwargs,
) -> Any:
    """
    Convert a collection/`Stream` of spaCy `Doc` objects (or already-converted
    dicts) into a `datasets.IterableDataset`.
    Examples
    --------
    1) Convert a `Stream` of HuggingFace NER examples into spaCy `Doc`s (reader),
       process them and create an `IterableDataset` of dictionaries using the
       `hf_ner` writer converter::

           import edsnlp

           stream = edsnlp.data.from_huggingface_dataset (
               "lhoestq/conll2003",
               split="train",
               converter="hf_ner",
           )

           # Apply a pipeline or other processing
           stream = stream.map_pipeline(nlp)

           # Export as HF IterableDataset of dicts (no push)
           hf_iter = edsnlp.data.to_huggingface_dataset (
               stream,
               converter="hf_ner",
           )

           )

    2) Convert plain text Docs to HF text-format dicts::

           edsnlp.data.to_huggingface_dataset (
               docs_stream,
               converter=("hf_text"),
               execute=True,
               # converter kwargs are validated and forwarded by
               # `get_doc2dict_converter` (e.g. `text_column`, `id_column`).
           )

    Parameters
    ----------
    data: Union[Any, Stream]
        Iterable of `Doc` objects or a `Stream`. If `converter` is provided the
        stream items are expected to be spaCy `Doc`s. Otherwise items should
        already be mapping-like dicts.
    converter: Optional[Union[str, Callable]]
        Converter name or callable used to transform `Doc` -> dict before
        creating the dataset. Typical values: ``"hf_ner_doc2dict"`` or
        ``"hf_text_doc2dict"``. Converter kwargs may be passed via ``**kwargs``.
    execute: bool
        If False, return a transformed `Stream` (not executed). If True (default)
        produce and return a `datasets.IterableDataset` (or pushed `Dataset`).
    **kwargs: dict
        Extra kwargs forwarded to the converter factory.

    Returns
    -------
    Union[datasets.IterableDataset, datasets.Dataset]
        An ``IterableDataset`` containing the converted data.
    """

    data = Stream.ensure_stream(data)

    if converter:
        conv, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(conv, kwargs=kwargs)

    if not execute:
        return data

    datasets = _import_datasets()

    # Pass a zero-arg callable that returns a generator function expected by HF
    gen_callable = lambda: _iter_from_stream(data)

    return datasets.IterableDataset.from_generator(gen_callable)