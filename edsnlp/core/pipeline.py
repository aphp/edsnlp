# ruff: noqa: E501
import functools
import os
import shutil
import warnings
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import spacy
import srsly
from confit import Config, validate_arguments
from confit.errors import ConfitValidationError, patch_errors
from confit.utils.collections import join_path, split_path
from confit.utils.xjson import Reference
from spacy.language import BaseDefaults
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import get_lang_class
from spacy.vocab import Vocab, create_vocab

from ..accelerators.base import Accelerator, FromDoc, ToDoc
from ..core.registry import PIPE_META, CurriedFactory, FactoryMeta, registry
from ..utils.collections import (
    FrozenDict,
    FrozenList,
    batch_compress_dict,
    decompress_dict,
    multi_tee,
)

if TYPE_CHECKING:
    import torch

import edsnlp

EMPTY_LIST = FrozenList()


class CacheEnum(str, Enum):
    preprocess = "preprocess"
    collate = "collate"
    forward = "forward"


Pipe = TypeVar("Pipe", bound=Callable[[Doc], Doc])


class Pipeline:
    """
    New pipeline to use as a drop-in replacement for spaCy's pipeline.
    It uses PyTorch as the deep-learning backend and allows components to share
    subcomponents.

    See the documentation for more details.
    """

    def __init__(
        self,
        lang: str,
        create_tokenizer: Callable[["Pipeline"], Optional[Tokenizer]] = None,
        vocab: Union[bool, Vocab] = True,
        batch_size: Optional[int] = 4,
        vocab_config: Type[BaseDefaults] = BaseDefaults,
        meta: Dict[str, Any] = None,
    ):
        """
        Parameters
        ----------
        lang: str
            Language code
        create_tokenizer: Callable[['Pipeline'], Optional[Tokenizer]]
            Function that creates a tokenizer for the pipeline
        vocab: Union[bool, Vocab]
            Whether to create a new vocab or use an existing one
        batch_size: Optional[int]
            Batch size to use in the `.pipe()` method
        vocab_config: Type[BaseDefaults]
            Configuration for the vocab
        meta: Dict[str, Any]
            Meta information about the pipeline
        """
        spacy_blank_cls = get_lang_class(lang)

        self.Defaults = spacy_blank_cls.Defaults
        self.batch_size = batch_size
        if (vocab is not True) == (vocab_config is not None):
            raise ValueError(
                "You must specify either vocab or vocab_config but not both."
            )
        if vocab is True:
            vocab = create_vocab(lang, vocab_config)

        self.vocab = vocab

        if create_tokenizer is None:
            create_tokenizer = Config.resolve(
                spacy_blank_cls.default_config.merge(spacy_blank_cls.Defaults.config)[
                    "nlp"
                ]["tokenizer"]
            )

        self._tokenizer_config = Config.serialize(create_tokenizer)
        self.tokenizer = create_tokenizer(self)
        self._components: List[Tuple[str, Pipe]] = []
        self._disabled: List[str] = []
        self._path: Optional[Path] = None
        self.meta = dict(meta) if meta is not None else {}
        self.lang: str = lang
        self._cache: Optional[Dict] = None
        self._cache_is_writeonly = False

    @property
    def pipeline(self) -> List[Tuple[str, Pipe]]:
        return FrozenList(self._components)

    @property
    def pipe_names(self) -> List[str]:
        return FrozenList([name for name, _ in self._components])

    component_names = pipe_names

    def get_pipe(self, name: str) -> Pipe:
        """
        Get a component by its name.

        Parameters
        ----------
        name: str
            The name of the component to get.

        Returns
        -------
        Pipe
        """
        for n, pipe in self.pipeline:
            if n == name:
                return pipe
        raise ValueError(f"Pipe {repr(name)} not found in pipeline.")

    def has_pipe(self, name: str) -> bool:
        """
        Check if a component exists in the pipeline.

        Parameters
        ----------
        name: str
            The name of the component to check.

        Returns
        -------
        bool
        """
        return any(n == name for n, _ in self.pipeline)

    def create_pipe(
        self,
        factory: str,
        name: str,
        config: Dict[str, Any] = None,
    ) -> Pipe:
        """
        Create a component from a factory name.

        Parameters
        ----------
        factory: str
            The name of the factory to use
        name: str
            The name of the component
        config: Dict[str, Any]
            The config to pass to the factory

        Returns
        -------
        Pipe
        """
        try:
            curried_factory: CurriedFactory = Config(
                {
                    "@factory": factory,
                    **(config if config is not None else {}),
                }
            ).resolve(registry=registry)
            pipe = curried_factory.instantiate(nlp=self, path=(name,))
        except ConfitValidationError as e:
            raise e.with_traceback(None)
        return pipe

    def add_pipe(
        self,
        factory: Union[str, Pipe],
        first: bool = False,
        before: Optional[str] = None,
        after: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Pipe:
        """
        Add a component to the pipeline.

        Parameters
        ----------
        factory: Union[str, Pipe]
            The name of the component to add or the component itself
        name: Optional[str]
            The name of the component. If not provided, the name of the component
            will be used if it has one (.name), otherwise the factory name will be used.
        first: bool
            Whether to add the component to the beginning of the pipeline. This argument
            is mutually exclusive with `before` and `after`.
        before: Optional[str]
            The name of the component to add the new component before. This argument is
            mutually exclusive with `after` and `first`.
        after: Optional[str]
            The name of the component to add the new component after. This argument is
            mutually exclusive with `before` and `first`.
        config: Dict[str, Any]
            The arguments to pass to the component factory.

            Note that instead of replacing arguments with the same keys, the config
            will be merged with the default config of the component. This means that
            you can override specific nested arguments without having to specify the
            entire config.

        Returns
        -------
        Pipe
            The component that was added to the pipeline.
        """
        if isinstance(factory, str):
            if name is None:
                name = factory
            pipe = self.create_pipe(factory, name, config)
        else:
            if config is not None:
                raise ValueError(
                    "Can't pass config or name with an instantiated component",
                )
            pipe = factory
            if hasattr(pipe, "name"):
                if name is not None and name != pipe.name:
                    raise ValueError(
                        "The provided name does not match the name of the component."
                    )
                else:
                    name = pipe.name
            else:
                if name is None:
                    raise ValueError(
                        "The component does not have a name, so you must provide one",
                    )
                pipe.name = name
        assert sum([before is not None, after is not None, first]) <= 1, (
            "You can only use one of before, after, or first",
        )
        insertion_idx = (
            0
            if first
            else self.pipe_names.index(before)
            if before is not None
            else self.pipe_names.index(after) + 1
            if after is not None
            else len(self._components)
        )
        self._components.insert(insertion_idx, (name, pipe))
        return pipe

    def get_pipe_meta(self, name: str) -> FactoryMeta:
        """
        Get the meta information for a component.

        Parameters
        ----------
        name: str
            The name of the component to get the meta for.

        Returns
        -------
        Dict[str, Any]
        """
        pipe = self.get_pipe(name)
        return PIPE_META.get(pipe, {})

    def make_doc(self, text: str) -> Doc:
        """
        Create a Doc from text.

        Parameters
        ----------
        text: str
            The text to create the Doc from.

        Returns
        -------
        Doc
        """
        return self.tokenizer(text)

    def _ensure_doc(self, text: Union[str, Doc]) -> Doc:
        """
        Ensure that the input is a Doc.

        Parameters
        ----------
        text: Union[str, Doc]
            The text to create the Doc from, or a Doc.

        Returns
        -------
        Doc
        """
        return text if isinstance(text, Doc) else self.make_doc(text)

    def __call__(self, text: Union[str, Doc]) -> Doc:
        """
        Apply each component successively on a document.

        Parameters
        ----------
        text: Union[str, Doc]
            The text to create the Doc from, or a Doc.

        Returns
        -------
        Doc
        """
        doc = self._ensure_doc(text)

        with self.cache():
            for name, pipe in self.pipeline:
                if name in self._disabled:
                    continue
                # This is a hack to get around the ambiguity
                # between the __call__ method of Pytorch modules
                # and the __call__ methods of spacy components
                if hasattr(pipe, "batch_process"):
                    doc = next(iter(pipe.batch_process([doc])))
                else:
                    doc = pipe(doc)

        return doc

    @validate_arguments
    def pipe(
        self,
        inputs: Any,
        batch_size: Optional[int] = None,
        *,
        accelerator: Optional[Union[str, Accelerator]] = None,
        to_doc: Optional[ToDoc] = None,
        from_doc: FromDoc = lambda doc: doc,
    ) -> Iterable[Doc]:
        """
        Process a stream of documents by applying each component successively on
        batches of documents.

        Parameters
        ----------
        inputs: Iterable[Union[str, Doc]]
            The inputs to create the Docs from, or Docs directly.
        batch_size: Optional[int]
            The batch size to use. If not provided, the batch size of the pipeline
            object will be used.
        accelerator: Optional[Union[str, Accelerator]]
            The accelerator to use for processing the documents. If not provided,
            the default accelerator will be used.
        to_doc: ToDoc
            The function to use to convert the inputs to PDFDoc objects. By default,
            the `content` field of the inputs will be used if dict-like objects are
            provided, otherwise the inputs will be passed directly to the pipeline.
        from_doc: FromDoc
            The function to use to convert the PDFDoc objects to outputs. By default,
            the PDFDoc objects will be returned directly.

        Returns
        -------
        Iterable[Doc]
        """

        if batch_size is None:
            batch_size = self.batch_size

        if accelerator is None:
            accelerator = {"@accelerator": "simple", "batch_size": batch_size}
        if isinstance(accelerator, str):
            accelerator = {"@accelerator": accelerator, "batch_size": batch_size}
        if isinstance(accelerator, dict):
            accelerator = Config(accelerator).resolve(registry=registry)

        kwargs = {
            "inputs": inputs,
            "nlp": self,
            "to_doc": to_doc,
            "from_doc": from_doc,
        }
        for k, v in list(kwargs.items()):
            if v is None:
                del kwargs[k]

        with self.train(False):
            return accelerator(**kwargs)

    @contextmanager
    def cache(self):
        """
        Enable caching for all (trainable) components in the pipeline
        """
        was_not_cached = self._cache is None
        if was_not_cached:
            self._cache = {}
        yield
        if was_not_cached:
            self._cache = None

    def torch_components(
        self, disable: Container[str] = ()
    ) -> Iterable[Tuple[str, "edsnlp.core.torch_component.TorchComponent"]]:
        """
        Yields components that are PyTorch modules.

        Parameters
        ----------
        disable: Container[str]
            The names of disabled components, which will be skipped.

        Returns
        -------
        Iterable[Tuple[str, 'edsnlp.core.torch_component.TorchComponent']]
        """
        for name, pipe in self.pipeline:
            if name not in disable and hasattr(pipe, "batch_process"):
                yield name, pipe

    def post_init(self, data: Iterable[Doc], exclude: Optional[Set] = None):
        """
        Completes the initialization of the pipeline by calling the post_init
        method of all components that have one.
        This is useful for components that need to see some data to build
        their vocabulary, for instance.

        Parameters
        ----------
        data: Iterable[Doc]
            The documents to use for initialization.
            Each component will not necessarily see all the data.
        exclude: Optional[Container]
            Components to exclude from post initialization on data
        """
        exclude = set() if exclude is None else set(exclude)
        data = multi_tee(data)
        for name, pipe in self._components:
            if hasattr(pipe, "post_init"):
                pipe.post_init(data, exclude=exclude)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any] = {},
        *,
        vocab: Union[Vocab, bool] = True,
        disable: Union[str, Iterable[str]] = EMPTY_LIST,
        enable: Union[str, Iterable[str]] = EMPTY_LIST,
        exclude: Union[str, Iterable[str]] = EMPTY_LIST,
        meta: Dict[str, Any] = FrozenDict(),
    ):
        """
        Create a pipeline from a config object

        Parameters
        ----------
        config: Config
            The config to use
        vocab: Union["spacy.Vocab", bool]
            The spaCy vocab to use. If True, a new vocab will be created
        disable: Union[str, Iterable[str]]
            Components to disable
        enable: Union[str, Iterable[str]]
            Components to enable
        exclude: Union[str, Iterable[str]]
            Components to exclude
        meta: Dict[str, Any]
            Metadata to add to the pipeline

        Returns
        -------
        Pipeline
        """
        root_config = config.copy()

        if "nlp" in config:
            if "components" in config and "components" not in config["nlp"]:
                config["nlp"]["components"] = Reference("components")
                config = config["nlp"]

        config = dict(Config(config).resolve(root=root_config))
        components = config.pop("components", {})
        pipeline = config.pop("pipeline", ())
        tokenizer = config.pop("tokenizer", None)
        disable = (config.pop("disabled", ()), disable)

        nlp = Pipeline(
            vocab=vocab,
            create_tokenizer=tokenizer,
            meta=meta,
            **config,
        )

        # Since components are actually resolved as curried factories,
        # we need to instantiate them here
        for name, component in components.items():
            if not isinstance(component, CurriedFactory):
                raise ValueError(
                    f"Component {repr(name)} is not instantiable. Please make sure "
                    "that you didn't forget to add a '@factory' key to the component "
                    "config."
                )

        try:
            components = CurriedFactory.instantiate(components, nlp=nlp)
        except ConfitValidationError as e:
            e = ConfitValidationError(
                e.raw_errors,
                model=cls,
                name=cls.__module__ + "." + cls.__qualname__,
            )
            e.raw_errors = patch_errors(e.raw_errors, ("components",))
            raise e

        for name in pipeline:
            if name in exclude:
                continue
            if name not in components:
                raise ValueError(f"Component {repr(name)} not found in config")
            nlp.add_pipe(components[name], name=name)

        # Set of components name if it's in the disable list
        # or if it's not in the enable list and the enable list is not empty
        nlp._disabled = [
            name
            for name in nlp.pipe_names
            if name in disable or (enable and name not in enable)
        ]
        return nlp

    @property
    def disabled(self):
        """
        The names of the disabled components
        """
        return FrozenList(self._disabled)

    @classmethod
    def __get_validators__(cls):
        """
        Pydantic validators generator
        """
        yield cls.validate

    @classmethod
    def validate(cls, v, config=None):
        """
        Pydantic validator, used in the `validate_arguments` decorated functions
        """
        if isinstance(v, dict):
            return cls.from_config(v)
        if not isinstance(v, cls):
            raise ValueError("input is not a Pipeline or config dict")
        return v

    def preprocess(self, doc: Doc, supervision: bool = False):
        """
        Run the preprocessing methods of each component in the pipeline
        on a document and returns a dictionary containing the results, with the
        component names as keys.

        Parameters
        ----------
        doc: InputT
            The document to preprocess
        supervision: bool
            Whether to include supervision information in the preprocessing

        Returns
        -------
        Dict[str, Any]
        """
        prep = {}
        with self.cache():
            if supervision:
                for name, component in self.pipeline:
                    prep_comp = (
                        component.preprocess_supervised(doc)
                        if hasattr(component, "preprocess_supervised")
                        else component.preprocess(doc)
                        if hasattr(component, "preprocess")
                        else None
                    )
                    if prep_comp is not None:
                        prep[name] = prep_comp

        return prep

    def preprocess_many(self, docs: Iterable[Doc], compress=True, supervision=True):
        """
        Runs the preprocessing methods of each component in the pipeline on
        a collection of documents and returns an iterable of dictionaries containing
        the results, with the component names as keys.

        Parameters
        ----------
        docs: Iterable[InputT]
        compress: bool
            Whether to deduplicate identical preprocessing outputs of the results
            if multiple documents share identical subcomponents. This step is required
            to enable the cache mechanism when training or running the pipeline over a
            tabular datasets such as pyarrow tables that do not store referential
            equality information.
        supervision: bool
            Whether to include supervision information in the preprocessing

        Returns
        -------
        Iterable[OutputT]
        """
        preprocessed = map(
            functools.partial(self.preprocess, supervision=supervision), docs
        )
        if compress:
            preprocessed = batch_compress_dict(preprocessed)
        return preprocessed

    def collate(
        self,
        batch: Dict[str, Any],
    ):
        """
        Collates a batch of preprocessed samples into a single (maybe nested)
        dictionary of tensors by calling the collate method of each component.

        Parameters
        ----------
        batch: Dict[str, Any]
            The batch of preprocessed samples
        device: Optional[torch.device]
            The device to move the tensors to before returning them

        Returns
        -------
        Dict[str, Any]
            The collated batch
        """
        batch = decompress_dict(batch)
        with self.cache():
            for name, component in self.pipeline:
                if name in batch:
                    component_inputs = batch[name]
                    batch[name] = component.collate(component_inputs)
        return batch

    def parameters(self):
        """Returns an iterator over the Pytorch parameters of the components in the
        pipeline"""
        return (p for n, p in self.named_parameters())

    def named_parameters(self):
        """Returns an iterator over the Pytorch parameters of the components in the
        pipeline"""
        seen = set()
        for name, component in self.pipeline:
            if hasattr(component, "named_parameters"):
                for param_name, param in component.named_parameters():
                    if param in seen:
                        continue
                    seen.add(param)
                    yield f"{name}.{param_name}", param

    def to(self, device: Union[str, Optional["torch.device"]] = None):  # noqa F821
        """Moves the pipeline to a given device"""
        for name, component in self.torch_components():
            component.to(device)
        return self

    def train(self, mode=True):
        """
        Enables training mode on pytorch modules

        Parameters
        ----------
        mode: bool
            Whether to enable training or not
        """

        class context:
            def __enter__(self):
                pass

            def __exit__(ctx_self, type, value, traceback):
                for name, proc in self.torch_components():
                    proc.train(was_training[name])

        was_training = {name: proc.training for name, proc in self.torch_components()}
        for name, proc in self.torch_components():
            proc.train(mode)

        return context()

    def to_disk(
        self, path: Union[str, Path], *, exclude: Optional[Set[str]] = None
    ) -> None:
        """
        Save the pipeline to a directory.

        Parameters
        ----------
        path: Union[str, Path]
            The path to the directory to save the pipeline to. Every component will be
            saved to separated subdirectories of this directory, except for tensors
            that will be saved to a shared files depending on the references between
            the components.
        exclude: Sequence[str]
            The names of the components, or attributes to exclude from the saving
            process.
        """
        exclude = set() if exclude is None else exclude
        tensor_exclude = set(exclude)

        def save_tensors(path: Path):
            import safetensors.torch

            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)
            tensors = defaultdict(list)
            tensor_to_group = defaultdict(list)
            for pipe_name, pipe in self.torch_components(disable=tensor_exclude):
                for key, tensor in pipe.state_dict(keep_vars=True).items():
                    full_key = join_path((pipe_name, *split_path(key)))
                    tensors[tensor].append(full_key)
                    tensor_to_group[tensor].append(pipe_name)
            group_to_tensors = defaultdict(set)
            for tensor, group in tensor_to_group.items():
                group_to_tensors["+".join(sorted(set(group)))].add(tensor)
            for group, group_tensors in group_to_tensors.items():
                sub_path = path / f"{group}.safetensors"
                tensor_dict = {
                    "+".join(tensors[p]): p
                    for p in {p.data_ptr(): p for p in group_tensors}.values()
                }
                safetensors.torch.save_file(tensor_dict, sub_path)

        path = Path(path) if isinstance(path, str) else path

        if os.path.exists(path) and not os.path.exists(path / "config.cfg"):
            raise Exception(
                "The directory already exists and doesn't appear to be a"
                "saved pipeline. Please erase it manually or choose a "
                "different directory."
            )
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

        if "tokenizer" not in exclude:
            self.tokenizer.to_disk(path / "tokenizer", exclude=["vocab"])
        if "meta" not in exclude:
            srsly.write_json(path / "meta.json", self.meta)
        if "config" not in exclude:
            self.config.to_disk(path / "config.cfg")
        if "vocab" not in exclude:
            self.vocab.to_disk(path / "vocab")

        for pipe_name, pipe in self._components:
            if hasattr(pipe, "to_disk") and pipe_name not in exclude:
                pipe.to_disk(path / pipe_name, exclude=exclude)

        if "tensors" not in exclude:
            save_tensors(path / "tensors")

    def from_disk(
        self,
        path: Union[str, Path],
        *,
        exclude: Optional[Union[str, Sequence[str]]] = None,
        device: Optional[Union[str, "torch.device"]] = "cpu",  # noqa F821
    ) -> "Pipeline":
        """
        Load the pipeline from a directory. Components will be updated in-place.

        Parameters
        ----------
        path: Union[str, Path]
            The path to the directory to load the pipeline from
        exclude: Optional[Union[str, Sequence[str]]]
            The names of the components, or attributes to exclude from the loading
            process.
        device: Optional[Union[str, "torch.device"]]
            Device to use when loading the tensors
        """

        def deserialize_meta(path: Path) -> None:
            if path.exists():
                data = srsly.read_json(path)
                self.meta.update(data)
                # self.meta always overrides meta["vectors"] with the metadata
                # from self.vocab.vectors, so set the name directly
                self.vocab.vectors.name = data.get("vectors", {}).get("name")

        def deserialize_vocab(path: Path) -> None:
            if path.exists():
                self.vocab.from_disk(path, exclude=exclude)

        def deserialize_tensors(path: Path):
            import safetensors.torch

            torch_components = dict(self.torch_components())
            for file_name in path.iterdir():
                pipe_names = file_name.stem.split("+")
                if any(pipe_name in torch_components for pipe_name in pipe_names):
                    # We only load tensors in one of the pipes since parameters
                    # are expected to be shared
                    pipe = torch_components[pipe_names[0]]
                    tensor_dict = {}
                    for keys, tensor in safetensors.torch.load_file(
                        file_name, device=device
                    ).items():
                        split_keys = [split_path(key) for key in keys.split("+")]
                        key = next(key for key in split_keys if key[0] == pipe_names[0])
                        tensor_dict[join_path(key[1:])] = tensor
                    # Non-strict because tensors of a given pipeline can be shared
                    # between multiple files
                    print(f"Loading tensors of {pipe_names[0]} from {file_name}")
                    extra_tensors = set(tensor_dict) - set(
                        pipe.state_dict(keep_vars=True).keys()
                    )
                    if extra_tensors:
                        warnings.warn(
                            f"{file_name} contains tensors that are not in the state"
                            f"dict of {pipe_names[0]}: {sorted(extra_tensors)}"
                        )
                    pipe.load_state_dict(tensor_dict, strict=False)

        exclude = (
            set()
            if exclude is None
            else {exclude}
            if isinstance(exclude, str)
            else set(exclude)
        )

        path = Path(path) if isinstance(path, str) else path
        if "meta" not in exclude:
            deserialize_meta(path / "meta.json")
        if (path / "vocab").exists() and "vocab" not in exclude:
            deserialize_vocab(path / "vocab")
        if "tokenizer" not in exclude:
            self.tokenizer.from_disk(path / "tokenizer", exclude=["vocab"])
        for name, proc in self._components:
            if hasattr(proc, "from_disk") and name not in exclude:
                proc.from_disk(path / name, exclude=exclude)
            # Convert to list here in case exclude is (default) tuple
            exclude.add(name)

        if "tensors" not in exclude:
            deserialize_tensors(path / "tensors")

        self._path = path  # type: ignore[assignment]
        return self

    # override config property getter to remove "factory" key from components
    @property
    def cfg(self) -> Config:
        """
        Returns the config of the pipeline, including the config of all components.
        Updated from spacy to allow references between components.
        """
        return Config(
            {
                "lang": self.lang,
                "pipeline": list(self.pipe_names),
                "components": {key: component for key, component in self._components},
                "disabled": list(self.disabled),
                "tokenizer": self._tokenizer_config,
            }
        )

    @property
    def config(self) -> Config:
        config = Config({"nlp": self.cfg.copy()})
        config["components"] = config["nlp"].pop("components")
        return config.serialize()

    def select_pipes(
        self,
        *,
        disable: Optional[Union[str, Iterable[str]]] = None,
        enable: Optional[Union[str, Iterable[str]]] = None,
    ):
        """
        Temporarily disable and enable components in the pipeline.

        Parameters
        ----------
        disable: Optional[Union[str, Iterable[str]]]
            The name of the component to disable, or a list of names.
        enable: Optional[Union[str, Iterable[str]]]
            The name of the component to enable, or a list of names.
        """

        class context:
            def __enter__(self):
                pass

            def __exit__(ctx_self, type, value, traceback):
                self._disabled = disabled_before

        if enable is None and disable is None:
            raise ValueError("Expected either `enable` or `disable`")
        disable = [disable] if isinstance(disable, str) else disable
        pipe_names = set(self.pipe_names)
        if enable is not None:
            enable = [enable] if isinstance(enable, str) else enable
            if set(enable) - pipe_names:
                raise ValueError(
                    "Enabled pipes {} not found in pipeline.".format(
                        sorted(set(enable) - pipe_names)
                    )
                )
            to_disable = [pipe for pipe in self.pipe_names if pipe not in enable]
            # raise an error if the enable and disable keywords are not consistent
            if disable is not None and disable != to_disable:
                raise ValueError("Inconsistent values for `enable` and `disable`")
            disable = to_disable

        if set(disable) - pipe_names:
            raise ValueError(
                "Disabled pipes {} not found in pipeline.".format(
                    sorted(set(disable) - pipe_names)
                )
            )

        disabled_before = self._disabled
        self._disabled = disable
        return context()


def blank(
    lang: str,
    config: Union[Dict[str, Any], Config] = {},
):
    """
    Loads an empty EDS-NLP Pipeline, similarly to `spacy.blank`. In addition to
    standard components, this pipeline supports EDS-NLP trainable torch components.

    Examples
    --------
    ```python
    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.covid")
    ```

    Parameters
    ----------
    lang: str
        Language ID, e.g. "en", "fr", "eds", etc.
    config: Union[Dict[str, Any], Config]
        The config to use for the pipeline

    Returns
    -------
    Pipeline
        The new empty pipeline instance.
    """
    # Check if language is registered / entry point is available
    config_lang = (
        config["nlp"].get("lang")
        if isinstance(config.get("nlp"), dict)
        else config.get("lang")
    )
    if config_lang is not None and config_lang != lang:
        raise ValueError(
            "The language specified in the config does not match the lang argument."
        )
    return Pipeline.from_config({"lang": lang, **config})


def load(
    config: Union[Path, str, Config],
    exclude: Optional[Union[str, Iterable[str]]] = None,
):
    """
    Load a pipeline from a config file or a directory.

    Parameters
    ----------
    config: Union[Path, str, Config]
        The config to use for the pipeline, or the path to a config file or a directory.
    exclude: Optional[Union[str, Iterable[str]]]
        The names of the components, or attributes to exclude from the loading
        process. :warning: The `exclude` argument will be mutated in place.

    Returns
    -------
    Pipeline
    """
    error = "The load function expects a Config or a path to a config file"
    if isinstance(config, (Path, str)):
        path = Path(config)
        if path.is_dir():
            path = Path(path) if isinstance(path, str) else path
            config = Config.from_disk(path / "config.cfg")
            nlp = Pipeline.from_config(config)
            nlp.from_disk(path, exclude=exclude)
            return nlp
        elif path.is_file():
            config = Config.from_disk(path)
        else:
            raise ValueError(error)
    elif not isinstance(config, Config):
        raise ValueError(error)

    return Pipeline.from_config(config)


PipelineProtocol = Union[Pipeline, spacy.Language]
