from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from spacy import Errors, Vocab, registry, util
from spacy.language import DEFAULT_CONFIG, FactoryMeta, Language
from spacy.pipe_analysis import validate_attrs
from spacy.tokens import Doc
from spacy.util import SimpleFrozenDict, SimpleFrozenList, raise_error
from spacy.vocab import create_vocab

from edsnlp.core.registries import accepted_arguments


@classmethod
def factory(
    cls,
    name: str,
    *,
    default_config: Dict[str, Any] = SimpleFrozenDict(),
    assigns: Iterable[str] = SimpleFrozenList(),
    requires: Iterable[str] = SimpleFrozenList(),
    retokenizes: bool = False,
    default_score_weights: Dict[str, Optional[float]] = SimpleFrozenDict(),
    func: Optional[Callable] = None,
) -> Callable:
    """
    Patched from spaCy to allow back dots in factory
    names (https://github.com/aphp/edsnlp/pull/152)

    Register a new pipeline component factory. Can be used as a decorator
    on a function or classmethod, or called as a function with the factory
    provided as the func keyword argument. To create a component and add
    it to the pipeline, you can use nlp.add_pipe(name).

    name (str): The name of the component factory.
    default_config (Dict[str, Any]): Default configuration, describing the
        default values of the factory arguments.
    assigns (Iterable[str]): Doc/Token attributes assigned by this component,
        e.g. "token.ent_id". Used for pipeline analysis.
    requires (Iterable[str]): Doc/Token attributes required by this component,
        e.g. "token.ent_id". Used for pipeline analysis.
    retokenizes (bool): Whether the component changes the tokenization.
        Used for pipeline analysis.
    default_score_weights (Dict[str, Optional[float]]): The scores to report during
        training, and their default weight towards the final score used to
        select the best model. Weights should sum to 1.0 per component and
        will be combined and normalized for the whole pipeline. If None,
        the score won't be shown in the logs or be weighted.
    func (Optional[Callable]): Factory function if not used as a decorator.

    DOCS: https://spacy.io/api/language#factory
    """
    if not isinstance(name, str):
        raise ValueError(Errors.E963.format(decorator="factory"))
    if not isinstance(default_config, dict):
        raise ValueError(
            Errors.E962.format(
                style="default config", name=name, cfg_type=type(default_config)
            )
        )

    def add_factory(factory_func: Callable) -> Callable:
        internal_name = cls.get_factory_name(name)
        if internal_name in registry.factories.namespace:
            # We only check for the internal name here – it's okay if it's a
            # subclass and the base class has a factory of the same name. We
            # also only raise if the function is different to prevent raising
            # if module is reloaded.
            existing_func = registry.factories.get(internal_name)
            if not util.is_same_func(factory_func, existing_func):
                raise ValueError(
                    Errors.E004.format(
                        name=name, func=existing_func, new_func=factory_func
                    )
                )

        util.get_arg_names(factory_func)
        if len(accepted_arguments(factory_func, ["nlp", "name"])) < 2:
            raise ValueError(Errors.E964.format(name=name))
        # Officially register the factory so we can later call
        # registry.resolve and refer to it in the config as
        # @factories = "spacy.Language.xyz". We use the class name here so
        # different classes can have different factories.
        registry.factories.register(internal_name, func=factory_func)
        factory_meta = FactoryMeta(
            factory=name,
            default_config=default_config,
            assigns=validate_attrs(assigns),
            requires=validate_attrs(requires),
            scores=list(default_score_weights.keys()),
            default_score_weights=default_score_weights,
            retokenizes=retokenizes,
        )
        cls.set_factory_meta(name, factory_meta)
        # We're overwriting the class attr with a frozen dict to handle
        # backwards-compat (writing to Language.factories directly). This
        # wouldn't work with an instance property and just produce a
        # confusing error – here we can show a custom error
        registry.factories.entry_points = False
        cls.factories = SimpleFrozenDict(
            registry.factories.get_all(), error=Errors.E957
        )
        registry.factories.entry_points = True
        return factory_func

    if func is not None:  # Support non-decorator use cases
        return add_factory(func)
    return add_factory


def __init__(
    self,
    vocab: Union[Vocab, bool] = True,
    *,
    max_length: int = 10**6,
    meta: Dict[str, Any] = {},
    create_tokenizer: Optional[Callable[["Language"], Callable[[str], Doc]]] = None,
    create_vectors: Optional[Callable[["Vocab"], Any]] = None,
    batch_size: int = 1000,
    **kwargs,
) -> None:  # pragma: no cover
    """
    EDS-NLP: Patched from spaCy do enable lazy-loading components

    Initialise a Language object.

    vocab (Vocab): A `Vocab` object. If `True`, a vocab is created.
    meta (dict): Custom meta data for the Language class. Is written to by
        models to add model meta data.
    max_length (int): Maximum number of characters in a single text. The
        current models may run out memory on extremely long texts, due to
        large internal allocations. You should segment these texts into
        meaningful units, e.g. paragraphs, subsections etc, before passing
        them to spaCy. Default maximum length is 1,000,000 charas (1mb). As
        a rule of thumb, if all pipeline components are enabled, spaCy's
        default models currently requires roughly 1GB of temporary memory per
        100,000 characters in one text.
    create_tokenizer (Callable): Function that takes the nlp object and
        returns a tokenizer.
    batch_size (int): Default batch size for pipe and evaluate.

    DOCS: https://spacy.io/api/language#init
    """

    # EDS-NLP: disable spacy default call to load every factory
    # since some of them may be missing dependencies (like torch)
    util.registry._entry_point_factories.get_all()

    self._config = DEFAULT_CONFIG.merge(self.default_config)
    self._meta = dict(meta)
    self._path = None
    self._optimizer = None
    # Component meta and configs are only needed on the instance
    self._pipe_meta: Dict[str, "FactoryMeta"] = {}  # meta by component
    self._pipe_configs = {}  # config by component

    if not isinstance(vocab, Vocab) and vocab is not True:
        raise ValueError(Errors.E918.format(vocab=vocab, vocab_type=type(Vocab)))
    if vocab is True:
        vectors_name = meta.get("vectors", {}).get("name")
        vocab = create_vocab(self.lang, self.Defaults, vectors_name=vectors_name)
        if (
            not create_vectors
            and "vectors" in self._config["nlp"]
            and "@vectors" in self._config["nlp"]["vectors"]
        ):
            vectors_cfg = {"vectors": self._config["nlp"]["vectors"]}
            create_vectors = registry.resolve(vectors_cfg)["vectors"]
        if create_vectors:
            vocab.vectors = create_vectors(vocab)
    else:
        if (self.lang and vocab.lang) and (self.lang != vocab.lang):
            raise ValueError(Errors.E150.format(nlp=self.lang, vocab=vocab.lang))
    self.vocab: Vocab = vocab
    if self.lang is None:
        self.lang = self.vocab.lang
    self._components: List[Tuple[str, Callable[[Doc], Doc]]] = []
    self._disabled: Set[str] = set()
    self.max_length = max_length
    # Create the default tokenizer from the default config
    if not create_tokenizer:
        tokenizer_cfg = {"tokenizer": self._config["nlp"]["tokenizer"]}
        create_tokenizer = registry.resolve(tokenizer_cfg)["tokenizer"]
    self.tokenizer = create_tokenizer(self)
    self.batch_size = batch_size
    self.default_error_handler = raise_error


Language.factory = factory
Language.__init__ = __init__
