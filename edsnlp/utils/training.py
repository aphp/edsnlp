import os
import sys
import tempfile
from enum import Enum
from itertools import islice
from pathlib import Path
from random import shuffle
from typing import IO, TYPE_CHECKING, Any, Callable, Optional, Union

import spacy
from rich_logger import RichTablePrinter
from spacy.errors import Errors, Warnings
from spacy.schemas import ConfigSchemaTraining
from spacy.tokens import Doc, DocBin
from spacy.training.loop import train as train_loop
from spacy.util import get_sourced_components, logger, registry, resolve_dot_names
from thinc.api import ConfigValidationError, fix_random_seed, set_gpu_allocator
from thinc.config import Config
from tqdm import tqdm

from edsnlp.connectors import BratConnector
from edsnlp.utils.merge_configs import merge_configs

if TYPE_CHECKING:
    from spacy.language import Language  # noqa: F401

__all__ = ["Config", "train", "DEFAULT_TRAIN_CONFIG"]

DEFAULT_TRAIN_CONFIG = Config().from_str(
    """
[system]
    gpu_allocator = null
    seed = 0

[paths]
    train = null
    dev = null
    raw = null
    init_tok2vec = null
    vectors = null

[corpora]
    [corpora.train]
        @readers = "spacy.Corpus.v1"
        path = ${paths.train}
        max_length = 0
        gold_preproc = false
        limit = 0
        augmenter = null

    [corpora.dev]
        @readers = "spacy.Corpus.v1"
        path = ${paths.dev}
        max_length = 0
        gold_preproc = false
        limit = 0
        augmenter = null

[training]
    train_corpus = "corpora.train"
    dev_corpus = "corpora.dev"
    seed = ${system.seed}
    gpu_allocator = ${system.gpu_allocator}
    dropout = 0.1
    accumulate_gradient = 1
    patience = 10000
    max_epochs = 0
    max_steps = 20000
    eval_frequency = 200
    frozen_components = []
    before_to_disk = null

    [training.batcher]
        @batchers = "spacy.batch_by_words.v1"
        discard_oversize = false
        tolerance = 0.2
        get_length = null

    [training.batcher.size]
        @schedules = "compounding.v1"
        start = 100
        stop = 1000
        compound = 1.001
        t = 0.0

    [training.logger]
        @loggers = "eds.RichLogger.v1"
        progress_bar = false

    [training.optimizer]
        @optimizers = "Adam.v1"
        beta1 = 0.9
        beta2 = 0.999
        L2_is_weight_decay = true
        L2 = 0.01
        grad_clip = 1.0
        use_averages = false
        eps = 0.00000001
        learn_rate = 0.001

[initialize]
    vectors = ${paths.vectors}
    init_tok2vec = ${paths.init_tok2vec}
    vocab_data = null
    lookups = null
    before_init = null
    after_init = null
""",
    interpolate=False,
)


class DataFormat(Enum):
    brat = "brat"
    standoff = "brat"
    spacy = "spacy"


def make_spacy_corpus_config(
    train_data: Union[str, list[Doc]],
    dev_data: Union[str, list[Doc], int, float],
    data_format: Optional[DataFormat],
    nlp: Optional[spacy.Language] = None,
    seed: int = 0,
    reader: str = "spacy.Corpus.v1",
):
    """
    Helper to create a spacy's corpus config from training and dev data by
    loading the documents accordingly and exporting the documents using spacy's DocBin.

    Parameters
    ----------
    train_data: Union[str, list[Doc]]
        The training data. Can be:
            - a list of spacy.Doc
            - a path to a given dataset
    dev_data: Union[str, list[Doc], int, float]
        The development data. Can be:
            - a list of spacy.Doc
            - a path to a given dataset
            - the number of documents to take from the training data
            - the fraction of documents to take from the training data
    data_format: Optional[DataFormat]
        Optional data format to determine how we should load the documents from the disk
    nlp: Optional[spacy.Language]
        Optional spacy model to load documents from non-spacy formats (like brat)
    seed: int
        The seed if we need to shuffle the data when splitting the dataset
    reader: str
        Which spacy reader to use when loading the data

    Returns
    -------
    Config
    """
    fix_random_seed(seed)
    train_docs = dev_docs = None

    if data_format is None:
        if isinstance(train_data, list):
            assert all(isinstance(doc, Doc) for doc in train_data)
            train_docs = train_data
        elif isinstance(train_data, (str, Path)) and train_data.endswith(".spacy"):
            data_format = DataFormat.spacy
        else:
            raise Exception()
    if data_format == DataFormat.brat:
        train_docs = list(BratConnector(train_data).brat2docs(nlp))
    elif data_format == DataFormat.spacy:
        if isinstance(dev_data, (float, int)):
            train_docs = DocBin().from_disk(train_data)
    elif train_docs is None:
        raise Exception()

    if isinstance(dev_data, (float, int)):
        if isinstance(dev_data, float):
            n_dev = int(len(train_docs)) * dev_data
        else:
            n_dev = dev_data
        shuffle(train_docs)
        dev_docs = train_docs[:n_dev]
        train_docs = train_docs[n_dev:]
    elif data_format == DataFormat.brat:
        dev_docs = list(BratConnector(dev_data).brat2docs(nlp))
    elif data_format == DataFormat.spacy:
        pass
    elif data_format is None:
        if isinstance(dev_data, list):
            assert all(isinstance(doc, Doc) for doc in dev_data)
            dev_docs = dev_data
        else:
            raise Exception()
    else:
        raise Exception()

    if data_format != "spacy" or isinstance(dev_data, (float, int)):
        tmp_path = Path(tempfile.mkdtemp())
        train_path = tmp_path / "train.spacy"
        dev_path = tmp_path / "dev.spacy"

        DocBin(docs=train_docs).to_disk(train_path)
        DocBin(docs=dev_docs).to_disk(dev_path)
    else:
        train_path = train_data
        dev_path = dev_data

    return Config().from_str(
        f"""
        [corpora]

        [corpora.train]
            @readers = {reader}
            path = {train_path}
            max_length = 0
            gold_preproc = false
            limit = 0
            augmenter = null

        [corpora.dev]
            @readers = {reader}
            path = {dev_path}
            max_length = 0
            gold_preproc = false
            limit = 0
            augmenter = null
    """
    )


def train(
    nlp: spacy.Language,
    output_path: Union[Path, str],
    config: Union[Config, dict],
    use_gpu: int = -1,
):
    """
    Training help to learn weight of trainable components in a pipeline.
    This function has been adapted from
    https://github.com/explosion/spaCy/blob/397197e/spacy/cli/train.py#L18

    Parameters
    ----------
    nlp: spacy.Language
        Spacy model to train
    output_path: Union[Path,str]
        Path to save the model
    config: Union[Config,dict]
        Optional config overrides
    use_gpu: bool
        Which gpu to use for training (-1 means CPU)
    """
    if "components" in config:
        raise ValueError(
            "Cannot update components config after the model has been " "instantiated."
        )

    output_path = Path(output_path)
    nlp.config = merge_configs(
        nlp.config, DEFAULT_TRAIN_CONFIG, config, remove_extra=False
    )

    config = nlp.config.interpolate()

    nlp.config = config
    if "seed" not in config["training"]:
        raise ValueError(Errors.E1015.format(value="[training] seed"))
    if "gpu_allocator" not in config["training"]:
        raise ValueError(Errors.E1015.format(value="[training] gpu_allocator"))
    if config["training"]["seed"] is not None:
        fix_random_seed(config["training"]["seed"])
    allocator = config["training"]["gpu_allocator"]
    if use_gpu >= 0 and allocator:
        set_gpu_allocator(allocator)

    # Use nlp config here before it's resolved to functions
    sourced = get_sourced_components(config)

    # ----------------------------- #
    # Resolve functions and classes #
    # ----------------------------- #
    # Resolve all training-relevant sections using the filled nlp config
    T = registry.resolve(config["training"], schema=ConfigSchemaTraining)
    dot_names = [T["train_corpus"], T["dev_corpus"]]
    if not isinstance(T["train_corpus"], str):
        raise ConfigValidationError(
            desc=Errors.E897.format(
                field="training.train_corpus", type=type(T["train_corpus"])
            )
        )
    if not isinstance(T["dev_corpus"], str):
        raise ConfigValidationError(
            desc=Errors.E897.format(
                field="training.dev_corpus", type=type(T["dev_corpus"])
            )
        )
    train_corpus, dev_corpus = resolve_dot_names(config, dot_names)
    optimizer = T["optimizer"]
    # Components that shouldn't be updated during training
    frozen_components = T["frozen_components"]
    # Sourced components that require resume_training
    resume_components = [p for p in sourced if p not in frozen_components]
    logger.info(f"Pipeline: {nlp.pipe_names}")
    if resume_components:
        with nlp.select_pipes(enable=resume_components):
            logger.info(f"Resuming training for: {resume_components}")
            nlp.resume_training(sgd=optimizer)
    # Make sure that listeners are defined before initializing further
    nlp._link_components()
    with nlp.select_pipes(disable=[*frozen_components, *resume_components]):
        if T["max_epochs"] == -1:
            sample_size = 100
            logger.debug(
                f"Due to streamed train corpus, using only first {sample_size} "
                f"examples for initialization. If necessary, provide all labels "
                f"in [initialize]. More info: https://spacy.io/api/cli#init_labels"
            )
            nlp.initialize(
                lambda: islice(train_corpus(nlp), sample_size), sgd=optimizer
            )
        else:
            nlp.initialize(lambda: train_corpus(nlp), sgd=optimizer)
        logger.info(f"Initialized pipeline components: {nlp.pipe_names}")
    # Detect components with listeners that are not frozen consistently
    for name, proc in nlp.pipeline:
        for listener in getattr(
            proc, "listening_components", []
        ):  # e.g. tok2vec/transformer
            # Don't warn about components not in the pipeline
            if listener not in nlp.pipe_names:
                continue
            if listener in frozen_components and name not in frozen_components:
                logger.warning(Warnings.W087.format(name=name, listener=listener))
            # We always check this regardless, in case user freezes tok2vec
            if listener not in frozen_components and name in frozen_components:
                if name not in T["annotating_components"]:
                    logger.warning(Warnings.W086.format(name=name, listener=listener))

    os.makedirs(output_path, exist_ok=True)
    train_loop(nlp, output_path)


@registry.loggers("eds.RichLogger.v1")
def console_logger(
    progress_bar: bool = False,
) -> Callable[
    [spacy.Language],
    tuple[Callable[[Optional[dict[str, Any]]], None], Callable[[], None]],
]:
    """
    A rich based logger that renders nicely in Jupyter notebooks and console

    Parameters
    ----------
    progress_bar: bool
        Whether to show a training progress bar or not

    Returns
    -------
    tuple[Callable[[Optional[dict[str, Any]]], None], Callable[[], None]]]
    """

    def setup_printer(
        nlp: "Language", stdout: IO = sys.stdout, stderr: IO = sys.stderr
    ) -> tuple[Callable[[Optional[dict[str, Any]]], None], Callable[[], None]]:

        # ensure that only trainable components are logged
        logged_pipes = [
            name
            for name, proc in nlp.pipeline
            if hasattr(proc, "is_trainable") and proc.is_trainable
        ]
        eval_frequency = nlp.config["training"]["eval_frequency"]
        score_weights = nlp.config["training"]["score_weights"]
        score_cols = [
            col for col, value in score_weights.items() if value is not None
        ] + ["speed"]

        fields = {"epoch": {}, "step": {}}
        for pipe in logged_pipes:
            fields[f"loss_{pipe}"] = {
                "format": "{0:.2f}",
                "name": f"Loss {pipe}".upper(),
                "goal": "lower_is_better",
            }
        for score, weight in score_weights.items():
            if score != "speed" and weight is not None:
                fields[score] = {
                    "format": "{0:.2f}",
                    "name": score.upper(),
                    "goal": "higher_is_better",
                }
        fields["speed"] = {"name": "WPS"}
        fields["duration"] = {"name": "DURATION"}
        table_printer = RichTablePrinter(fields=fields)

        progress: Optional[tqdm] = None
        last_seconds = 0

        def log_step(info: Optional[dict[str, Any]]) -> None:
            nonlocal progress, last_seconds

            if info is None:
                # If we don't have a new checkpoint, just return.
                if progress is not None:
                    progress.update(1)
                return

            data = {
                "epoch": info["epoch"],
                "step": info["step"],
            }

            for pipe in logged_pipes:
                data[f"loss_{pipe}"] = float(info["losses"][pipe])

            for col in score_cols:
                score = info["other_scores"].get(col, 0.0)
                try:
                    score = float(score)
                except TypeError:
                    err = Errors.E916.format(name=col, score_type=type(score))
                    raise ValueError(err) from None
                if col != "speed":
                    score *= 100
                data[col] = score
            data["duration"] = info["seconds"] - last_seconds
            last_seconds = info["seconds"]

            if progress is not None:
                progress.close()
            table_printer.log(data)

            if progress_bar:
                # Set disable=None, so that it disables on non-TTY
                progress = tqdm(
                    total=eval_frequency, disable=None, leave=False, file=stderr
                )
                progress.set_description(f"Epoch {info['epoch'] + 1}")

        def finalize() -> None:
            table_printer.finalize()

        return log_step, finalize

    return setup_printer
