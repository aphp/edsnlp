import json
import math
import os
import time
import warnings
from collections import defaultdict
from contextlib import nullcontext
from itertools import chain
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

import torch
from accelerate import Accelerator, PartialState
from accelerate.tracking import GeneralTracker
from accelerate.utils import gather_object
from confit import Draft, validate_arguments
from confit.utils.random import set_seed
from tqdm import tqdm, trange
from typing_extensions import Literal

import edsnlp
from edsnlp import Pipeline, registry
from edsnlp.core.stream import Stream
from edsnlp.metrics.ner import NerMetric
from edsnlp.metrics.span_attribute import SpanAttributeMetric
from edsnlp.pipes.base import (
    BaseNERComponent,
    BaseRelationDetectorComponent,
    BaseSpanAttributeClassifierComponent,
)
from edsnlp.utils.batching import BatchSizeArg, stat_batchify
from edsnlp.utils.bindings import BINDING_SETTERS
from edsnlp.utils.collections import (
    chain_zip,
    decompress_dict,
    flatten,
    flatten_once,
    ld_to_dl,
)
from edsnlp.utils.span_getters import get_spans
from edsnlp.utils.typing import AsList

from ..core.torch_component import TorchComponent
from ..metrics.relations import RelationsMetric
from .optimizer import LinearSchedule, ScheduledOptimizer


def deep_add_flat(x, result, path=(), check_stats_key=True):
    if result is None:
        result = {}
    if isinstance(x, dict):
        for k, v in x.items():
            deep_add_flat(v, result, (*path, k), check_stats_key=check_stats_key)
        return result
    if not check_stats_key or "stats" in path and "__batch_hash__" not in path[-1]:
        if isinstance(x, (float, int)):
            path = "/".join(path)
            result[path] = result.get(path, 0) + x
        elif isinstance(x, torch.Tensor) and x.ndim == 0:
            path = "/".join(path)
            result[path] = result.get(path, 0) + x.item()
    return result


def deep_set_unflat(x, stats):
    for k, v in stats.items():
        path = k.split("/")
        current = x
        for p in path[:-1]:
            if p not in current:
                break
            current = current[p]
        else:
            current[path[-1]] = v
    return x


@validate_arguments
class GenericScorer:
    def __init__(
        self,
        batch_size: Union[int, str] = 1,
        autocast: Union[bool, Any] = None,
        speed: bool = True,
        **metrics,
    ):
        """
        A scorer to evaluate the model performance on various tasks.

        Parameters
        ----------
        batch_size: Union[int, str]
            The batch size to use for scoring. Can be an int (number of documents)
            or a string (batching expression like "2000 words").
        speed: bool
            Whether to compute the model speed (words/documents per second)
        autocast: Union[bool, Any]
            Whether to use autocasting for mixed precision during the evaluation,
            defaults to True.
        metrics: Dict[str, Any]
            A keyword arguments mapping of metric names to metrics objects. See the
            [metrics](/metrics) documentation for more info.
        """
        self.metrics = metrics
        self.speed = speed
        self.batch_size = batch_size
        self.autocast = autocast

    def __call__(self, nlp: Pipeline, docs: Iterable[Any]):
        scores = {}
        docs = list(docs)
        metrics = dict(self.metrics)

        # Speed
        if self.speed:
            t0 = time.time()
            list(
                nlp.pipe(
                    d.copy() for d in tqdm(docs, desc="Computing model speed")
                ).set_processing(
                    batch_size=self.batch_size,
                    autocast=self.autocast,
                )
            )
            duration = time.time() - t0
            scores["speed"] = dict(
                wps=sum(len(d) for d in docs) / duration,
                dps=len(docs) / duration,
            )

        # NER
        ner_pipes = [
            name for name, pipe in nlp.pipeline if isinstance(pipe, BaseNERComponent)
        ]
        ner_metrics = {
            name: metrics.pop(name)
            for name in list(metrics)
            if isinstance(metrics[name], NerMetric)
        }
        if ner_pipes and ner_metrics:
            clean_ner_docs = [d.copy() for d in tqdm(docs, desc="Copying docs")]
            for d in clean_ner_docs:
                d.ents = []
                d.spans.clear()
            with nlp.select_pipes(enable=ner_pipes):
                ner_preds = list(
                    nlp.pipe(tqdm(clean_ner_docs, desc="Predicting")).set_processing(
                        batch_size=self.batch_size,
                        autocast=self.autocast,
                    )
                )
            for name, metric in ner_metrics.items():
                scores[name] = metric(docs, ner_preds)

        # Qualification
        qlf_pipes = [
            name
            for name, pipe in nlp.pipeline
            if isinstance(pipe, BaseSpanAttributeClassifierComponent)
        ]
        span_attr_metrics = {
            name: metrics.pop(name)
            for name in list(metrics)
            if isinstance(metrics[name], SpanAttributeMetric)
        }
        if qlf_pipes and span_attr_metrics:
            clean_qlf_docs = [d.copy() for d in tqdm(docs, desc="Copying docs")]
            for doc in clean_qlf_docs:
                for name in qlf_pipes:
                    pipe = nlp.get_pipe(name)
                    for span in get_spans(doc, pipe.span_getter):
                        for qlf in nlp.get_pipe(name).attributes:
                            BINDING_SETTERS[(qlf, None)](span)
            with nlp.select_pipes(disable=ner_pipes):
                qlf_preds = list(
                    nlp.pipe(tqdm(clean_qlf_docs, desc="Predicting")).set_processing(
                        batch_size=self.batch_size,
                        autocast=self.autocast,
                    )
                )
            for name, metric in span_attr_metrics.items():
                scores[name] = metric(docs, qlf_preds)

        # Relations
        rel_pipes = [
            name
            for name, pipe in nlp.pipeline
            if isinstance(pipe, BaseRelationDetectorComponent)
        ]
        rel_metrics: Dict[str, RelationsMetric] = {  # type: ignore
            name: metrics.pop(name)
            for name in list(metrics)
            if isinstance(metrics[name], RelationsMetric)
        }
        if rel_pipes and rel_metrics:
            clean_rel_docs = [d.copy() for d in tqdm(docs, desc="Copying docs")]
            for doc in clean_rel_docs:
                for name in rel_pipes:
                    pipe: BaseRelationDetectorComponent = nlp.get_pipe(name)  # type: ignore
                    for candidate_getter in pipe.candidate_getter:
                        for span in (
                            *get_spans(doc, candidate_getter["head"]),
                            *get_spans(doc, candidate_getter["tail"]),
                        ):
                            for label in pipe.labels:
                                if label in span._.rel:
                                    span._.rel[label].clear()
            with nlp.select_pipes(disable=ner_pipes):
                rel_preds = list(nlp.pipe(tqdm(clean_rel_docs, desc="Predicting")))
            for name, scorer in rel_metrics.items():
                scores[name] = scorer(docs, rel_preds)

        # Custom metrics
        for name, metric in metrics.items():
            pred_docs = [d.copy() for d in tqdm(docs, desc="Copying docs")]
            preds = list(
                nlp.pipe(tqdm(pred_docs, desc="Predicting")).set_processing(
                    batch_size=self.batch_size,
                    autocast=self.autocast,
                )
            )
            scores[name] = metric(docs, preds)

        return scores


if TYPE_CHECKING:
    GenericScorer = Union[GenericScorer, Dict]


def ewm_moments(x, window, adjust=True, bias=False, state=None):
    if state is None:
        alpha = 2.0 / (window + 1)
        decay = 1 - alpha
        fresh_weight = 1 if adjust else alpha
        mean_val = x
        var_val = 0.0
        sum_w = 1.0
        sum_w2 = 1.0
        old_w = 1.0
        return (
            mean_val,
            float("nan"),
            [decay, fresh_weight, mean_val, var_val, sum_w, sum_w2, old_w],
        )
    else:
        decay, fresh_weight, mean_val, var_val, sum_w, sum_w2, old_w = state

    sum_w *= decay
    sum_w2 *= decay * decay
    old_w *= decay
    old_m = mean_val
    denom = old_w + fresh_weight
    mean_val = (old_w * old_m + fresh_weight * x) / denom
    d1 = old_m - mean_val
    d2 = x - mean_val
    var_val = (old_w * (var_val + d1 * d1) + fresh_weight * d2 * d2) / denom
    sum_w += fresh_weight
    sum_w2 += fresh_weight * fresh_weight
    old_w += fresh_weight

    state = [decay, fresh_weight, mean_val, var_val, sum_w, sum_w2, old_w]

    if not bias:
        num = sum_w * sum_w
        den = num - sum_w2
        var_val = var_val * (num / den) if den > 0 else float("nan")

    return mean_val, var_val, state


def default_optim(
    trained_pipes,
    *,
    task_lr: float = 3e-4,
    transformer_lr: float = 5e-5,
    warmup_rate: float = 0.1,
    max_steps: int,
):
    from edsnlp.pipes.trainable.embeddings.transformer.transformer import Transformer

    trf_pipe = next(
        (
            module
            for pipe in trained_pipes
            for name, module in pipe.named_component_modules()
            if isinstance(module, Transformer)
        ),
        None,
    )
    params = set(p for pipe in trained_pipes for p in pipe.parameters())
    trf_params = params & set(trf_pipe.parameters() if trf_pipe else ())

    return ScheduledOptimizer(
        torch.optim.AdamW(
            [
                {
                    "params": list(params - trf_params),
                    "lr": task_lr,
                    "schedules": LinearSchedule(
                        total_steps=max_steps,
                        warmup_rate=warmup_rate,
                        start_value=task_lr,
                    ),
                }
            ]
            + [
                {
                    "params": list(trf_params),
                    "lr": transformer_lr,
                    "schedules": LinearSchedule(
                        total_steps=max_steps,
                        warmup_rate=warmup_rate,
                        start_value=0,
                    ),
                },
            ][: 1 if transformer_lr else 0]
        )
    )


@validate_arguments
class TrainingData:
    def __init__(
        self,
        data: Stream,
        batch_size: BatchSizeArg,
        shuffle: Union[str, Literal[False]],
        sub_batch_size: Optional[BatchSizeArg] = None,
        pipe_names: Optional[AsList[str]] = None,
        post_init: bool = True,
    ):
        """
        A training data object.

        Parameters
        ----------
        data: Stream
            The stream of documents to train on. The documents will be
            preprocessed and collated according to the pipeline's components.
        batch_size: BatchSizeArg
            The batch size. Can be a batching expression like "2000 words",
            an int (number of documents), or a tuple (batch_size, batch_by).
            The batch_by argument should be a statistic produced by the
            pipes that will be trained. For instance, the `eds.span_pooler`
            component produces a "spans" statistic, that can be used to
            produce batches of no more than 16 spans by setting batch_size
            to "16 spans".
        shuffle: str
            The shuffle strategy. Can be "dataset" to shuffle the entire
            dataset (this can be memory-intensive for large file based
            datasets), "fragment" to shuffle the fragment-based datasets
            like parquet files, or a batching expression like "2000 words"
            to shuffle the dataset in chunks of 2000 words.
        sub_batch_size: Optional[BatchSizeArg]
            How to split each batch into sub-batches that will be fed to
            the model independently to accumulate gradients over.
            To split a batch of 8000 tokens into smaller batches of 1000
            tokens each, just set this to "1000 tokens".

            You can also request a number of splits, like "4 splits",
            to split the batch into N parts each close to (but less than)
            batch_size / N.
        pipe_names: Optional[Collection[str]]
            The names of the pipes that should be trained on this data.
            If None, defaults to all trainable pipes.
        post_init: bool
            Whether to call the pipeline's post_init method with the data
            before training.
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sub_batch_size = sub_batch_size
        self.pipe_names = set(pipe_names) if pipe_names else None
        self.post_init = post_init

    def __call__(self, nlp, device):
        data = self.data.loop()
        if self.shuffle:
            data = data.shuffle(self.shuffle)

        with nlp.select_pipes(enable=self.pipe_names):
            data = data.map(nlp.preprocess, kwargs=dict(supervision=True))
        batcher = stat_batchify(self.batch_size[1] or "docs")
        if self.sub_batch_size and self.sub_batch_size[1] == "splits":
            data = data.batchify(
                batch_size=self.batch_size[0] // self.sub_batch_size[0],
                batch_by=batcher,
            )
            data = data.batchify(batch_size=self.sub_batch_size[0])
            data = data.map(lambda b: [nlp.collate(sb, device=device) for sb in b])
        elif self.sub_batch_size:
            data = data.batchify(batch_size=self.batch_size[0], batch_by=batcher)
            sub_batcher = stat_batchify(self.sub_batch_size[1] or "docs")
            data = data.map(
                lambda batch: [
                    nlp.collate(sub_batch, device=device)
                    for sub_batch in sub_batcher(batch, self.sub_batch_size[0])
                ]
            )
        else:
            data = data.batchify(batch_size=self.batch_size[0], batch_by=batcher)
            data = data.map(nlp.collate, kwargs=dict(device=device))
        return data


class PipeDict(torch.nn.ModuleDict):
    def __init__(self, pipes, loss_scales):
        super().__init__(pipes)
        self.loss_scales = loss_scales

    def forward(self, batch, enable: Optional[Sequence[str]] = None):
        loss = None
        all_results = {}
        for name, pipe in self.items():
            if enable is None or name in enable:
                res = pipe(batch[name])
                all_results[name] = res
                if "loss" in res:
                    res["loss"] = res["loss"] * self.loss_scales.get(name, 1)
                    loss = res["loss"] if loss is None else loss + res["loss"]
                    if torch.isnan(loss):
                        raise ValueError(f"NaN loss at component {name}")
        return all_results, loss


def get_logger(
    logger: Union[bool, AsList[Union[str, Draft[GeneralTracker], GeneralTracker]]],
    project_name,
    logging_dir,
    **kwargs,
) -> List[GeneralTracker]:
    logger = ["rich", "json"] if logger is True else [] if not logger else logger
    logger_drafts_maybe = []
    for log in logger:
        if isinstance(log, str):
            cls = edsnlp.registry.loggers.get(log)
            logger_drafts_maybe.append(Draft(cls, {}))
        else:
            logger_drafts_maybe.append(log)
    logger = logger_drafts_maybe
    logger = [
        Draft.instantiate(obj, project_name=project_name, logging_dir=logging_dir)
        for obj in logger
    ]
    return logger


@validate_arguments(registry=registry)
def train(
    *,
    nlp: Pipeline,
    train_data: AsList[TrainingData],
    val_data: AsList[Stream] = [],
    seed: int = 42,
    max_steps: int = 1000,
    optimizer: Union[
        Draft[ScheduledOptimizer],
        ScheduledOptimizer,
        torch.optim.Optimizer,
    ] = None,
    validation_interval: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
    grad_max_norm: float = 5.0,
    grad_ewm_window: int = 100,
    grad_dev_policy: Optional[Literal["clip_mean", "clip_threshold", "skip"]] = None,
    grad_max_dev: float = 7.0,
    loss_scales: Dict[str, float] = {},
    scorer: GenericScorer = GenericScorer(),
    num_workers: int = 0,
    cpu: bool = False,
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = "no",
    output_dir: Union[Path, str] = Path("artifacts"),
    output_model_dir: Optional[Union[Path, str]] = None,
    save_model: bool = True,
    logger: Union[bool, AsList[Union[str, GeneralTracker, Draft[GeneralTracker]]]] = True,  # noqa: E501
    log_weight_grads: bool = False,
    on_validation_callback: Optional[Callable[[Dict], None]] = None,
    project_name: str = None,
    config_meta: Optional[Dict] = None,
    **kwargs,
):  # fmt: skip
    """
    Train a pipeline.

    Parameters
    ----------
    nlp: Pipeline
        The pipeline that will be trained in place.
    train_data: AsList[TrainingData]
        The training data. Can be a single
        [TrainingData][edsnlp.training.trainer.TrainingData] object, a dict that
        will be cast or a list of these objects.

        ??? note "`TrainingData` object/dictionary"
            ::: edsnlp.training.trainer.TrainingData
                options:
                    heading_level: 1
                    only_parameters: "no-header"
                    skip_parameters: []
                    show_source: false
                    show_toc: false
    val_data: AsList[Stream]
        The validation data. Can be a single Stream object or a list of
        Stream.
    seed: int
        The random seed
    max_steps: int
        The maximum number of training steps
    optimizer: Union[ScheduledOptimizer, Draft[ScheduledOptimizer], torch.optim.Optimizer]
        The optimizer. If None, a default optimizer will be used.

        ??? note "`ScheduledOptimizer` object/dictionary"
            ::: edsnlp.training.optimizer.ScheduledOptimizer
                options:
                    heading_level: 1
                    only_parameters: "no-header"
                    skip_parameters: []
                    show_source: false
                    show_toc: false
    validation_interval: Optional[int]
        The number of steps between each evaluation. Defaults to 1/10 of max_steps
    checkpoint_interval: Optional[int]
        The number of steps between each model save. Defaults to validation_interval
    grad_max_norm: float
        The maximum gradient norm
    grad_dev_policy: Optional[Literal["clip_mean", "clip_threshold"]]
        The policy to apply when a gradient spike is detected, ie. when the
        gradient norm is higher than the mean + std * grad_max_dev. Can be:

        - "clip_mean": clip the gradients to the mean gradient norm
        - "clip_threshold": clip the gradients to the mean + std * grad_max_dev
        - "skip": skip the step

        These do not apply to `grad_max_norm` that is always enforced when it is not
        None, since `grad_max_norm` is not adaptive and would most likely prohibit
        the model from learning during the early stages of training when gradients are
        expected to be high.
    grad_ewm_window: int
        Approximately how many steps should we look back to compute the average
        gradient norm and variance to detect gradient deviation spikes.
    grad_max_dev: float
        The threshold to apply to detect gradient spikes. A spike is detected
        when the value is higher than the mean + variance * threshold.
    loss_scales: Dict[str, float]
        The loss scales for each component (useful for multi-task learning)
    scorer: GenericScorer
        How to score the model. Expects a `GenericScorer` object or a dict
        containing a mapping of metric names to metric objects.

        ??? note "`GenericScorer` object/dictionary"
            ::: edsnlp.training.trainer.GenericScorer
                options:
                    heading_level: 1
                    only_parameters: "no-header"
                    skip_parameters: []
                    show_source: false
                    show_toc: false
    num_workers: int
        The number of workers to use for preprocessing the data in parallel.
        Setting it to 0 means no parallelization : data is processed on the
        main thread which may induce latency slow down the training. To
        avoid this, a good practice consist in doing the preprocessing either
        before training or in parallel in a separate process. Because of how
        EDS-NLP handles stream multiprocessing, changing this value
        will affect the order of the documents in the produces batches.
        A stream [1, 2, 3, 4, 5, 6] split in batches of size 3 will produce:

        - [1, 2, 3] and [4, 5, 6] with 1 worker
        - [1, 3, 5] and [2, 4, 6] with 2 workers
    cpu: bool
        Whether to use force training on CPU. On MacOS, this might be
        necessary to get around some `mps` backend issues.
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"]
        The mixed precision mode. Can be "no", "fp16", "bf16" or "fp8".
    output_dir: Union[Path, str]
        The output directory, which will contain a `model-last` directory
        with the last model, and a `train_metrics.json` file with the
        training metrics and stats.
    output_model_dir: Optional[Union[Path, str]]
        The directory where to save the model. If None, defaults to
        `output_dir / "model-last"`.
    save_model: bool
        Whether to save the model or not. This can be useful if you are only
        interested in the metrics, but no the model, and want to avoid
        spending time dumping the model weights to the disk.
    logger: Union[bool, AsList[Union[str, Partial[GeneralTracker], GeneralTracker]]]
        The logger to use. Can be a boolean to use the default loggers (rich
        and json), a list of logger names, or a list of logger objects.

        You can use huggingface accelerate integrated loggers (`tensorboard`,
        `wandb`, `comet_ml`, `aim`, `mlflow`, `clearml`, `dvclive`), or
        EDS-NLP simple loggers, or a combination of both:

        - `csv`: logs to a CSV file in `output_dir` (`artifacts/metrics.csv`)
        - `json`: logs to a JSON file in `output_dir` (`artifacts/metrics.json`)
        - `rich`: logs to a rich table in the terminal
    log_weight_grads: bool
        Whether to log the weight gradients during training.
    on_validation_callback: Optional[Callable[[Dict], None]]
        A callback function invoked during validation steps to handle custom logic.
    project_name: str
        The project name, used to group experiments in some loggers. If None,
        defaults to the path of the config file, relative to the home directory, with
        slashes replaced by double underscores.
    kwargs: Dict
        Additional keyword arguments.

    Returns
    -------
    Pipeline
        The trained pipeline
    """  # noqa: E501
    # hack to ensure cpu is set before the accelerator is indirectly initialized
    # when creating the trackers
    PartialState(cpu=cpu)
    project_name = project_name or str(
        os.curdir if config_meta is None else config_meta["config_path"][0]
    ).replace("/", "__")
    output_dir = Path(output_dir or Path.cwd() / "artifacts")
    output_model_dir = Path(output_model_dir or output_dir / "model-last")
    accelerator = Accelerator(
        cpu=cpu,
        mixed_precision=mixed_precision,
        log_with=get_logger(
            logger,
            # default project name, the user can override this when creating the logger
            project_name=project_name,
            logging_dir=output_dir,
        ),
    )
    # accelerator.register_for_checkpointing(dataset)
    is_main_process = accelerator.is_main_process
    device = accelerator.device

    if "max_grad_norm" in kwargs:
        warnings.warn(
            "The 'max_grad_norm' argument is deprecated. Use 'grad_max_norm' instead."
        )
        grad_max_norm = kwargs.pop("max_grad_norm")

    unresolved_config = None
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_model_dir, exist_ok=True)
        if config_meta is not None:  # pragma: no cover
            unresolved_config = config_meta["unresolved_config"]
            unresolved_config["train"]["project_name"] = project_name
            print(unresolved_config.to_yaml_str())
            unresolved_config.to_disk(output_dir / "train_config.yml")
        # TODO: handle config_meta is None
    accelerator.init_trackers(
        project_name,
        config=json.loads(json.dumps(unresolved_config, default=str)),
    )  # in theory project name shouldn't be used

    validation_interval = validation_interval or max_steps // 10
    checkpoint_interval = checkpoint_interval or validation_interval
    trainable_pipe_names = {name for name, pipe in nlp.torch_components()}
    phases = nlp.connected_pipes_names()
    accelerator.print("Trainable components: " + ", ".join(trainable_pipe_names))
    accelerator.print(
        "Training phases:"
        + "".join(f"\n - {i + 1}: {', '.join(n)}" for i, n in enumerate(phases))
    )

    if kwargs:
        raise ValueError(f"Unknown arguments: {', '.join(kwargs)}")

    # Prepare validation docs
    val_docs = list(chain.from_iterable(val_data))

    # Initialize pipeline with training documents
    nlp.post_init(chain_zip([td.data for td in train_data if td.post_init]))

    all_params = set(nlp.parameters())
    optim = optimizer
    del optimizer
    optim = Draft.instantiate(optim, module=nlp, total_steps=max_steps)
    if optim is None:
        warnings.warn(
            "No optimizer provided, using default optimizer with default parameters"
        )
        optim = default_optim(
            [nlp.get_pipe(name) for name in trainable_pipe_names],
            max_steps=max_steps,
            **{
                k: kwargs.pop(k)
                for k in ("task_lr", "transformer_lr", "warmup_rate")
                if k in kwargs
            },
        )
    optim: torch.nn.Optimizer = Draft.instantiate(
        optim,
        module=nlp,
        total_steps=max_steps,
    )

    for td in train_data:
        if not (td.pipe_names is None or td.pipe_names <= trainable_pipe_names):
            raise ValueError(
                f"Training data pipe names {td.pipe_names} should be a subset of "
                f"the trainable pipe names {trainable_pipe_names}, or left to None "
                f"use this dataset for all trainable components."
            )

    for phase_i, pipe_names in enumerate(phases):
        trained_pipes_local: Dict[str, TorchComponent] = {
            n: nlp.get_pipe(n) for n in pipe_names
        }
        trained_pipes = PipeDict(trained_pipes_local, loss_scales)
        trained_pipes_params = set(trained_pipes.parameters())
        phase_training_data = [
            td
            for td in train_data
            if td.pipe_names is None or set(td.pipe_names) & set(pipe_names)
        ]

        if len(phase_training_data) == 0:
            raise ValueError(
                f"No training data found for phase {phase_i + 1} with components "
                f"{', '.join(pipe_names)}. Make sure that these components are "
                f"listed in the 'pipe_names' attribute of at least one of the "
                f"provided training data."
            )

        with nlp.select_pipes(disable=trainable_pipe_names - set(pipe_names)):
            accelerator.print(f"Phase {phase_i + 1}: training {', '.join(pipe_names)}")
            set_seed(seed)

            optim_params = {p for g in optim.param_groups for p in g["params"]}
            grad_params = set()
            for param in all_params:
                has_grad_param = param in optim_params and param in trained_pipes_params
                if has_grad_param:
                    grad_params.add(param)
                param.requires_grad_(has_grad_param)

            accelerator.print("Optimizing groups:")
            for g in optim.param_groups:
                accelerator.print(
                    " - {} weight tensors ({:,} parameters){}".format(
                        len([p for p in g["params"] if p in grad_params]),
                        sum([p.numel() for p in g["params"] if p in grad_params]),
                        ": " + " & ".join(g.get("selectors", "*"))
                        if "selectors" in g
                        else "",
                    )
                )
            accelerator.print(
                f"Keeping frozen {len(all_params - grad_params):} weight tensors "
                f"({sum(p.numel() for p in all_params - grad_params):,} parameters)"
            )

            nlp.train(True)

            phase_datasets = [
                td(nlp, device).set_processing(
                    num_cpu_workers=num_workers,
                    process_start_method="spawn",
                )
                for td in phase_training_data
            ]
            iterator = iter(zip(*(phase_datasets)))
            (accel_optim, trained_pipes) = accelerator.prepare(optim, trained_pipes)
            if hasattr(accel_optim.optimizer, "initialize"):
                accel_optim.optimizer.initialize()

            ewm_state = grad_mean = grad_var = None
            count = 0
            spikes = 0
            flat_cum_stats = defaultdict(float)
            flat_cum_res = defaultdict(float)
            flat_cum_weight_metrics = defaultdict(float)
            set_seed(seed)
            # Training loop
            for step in trange(
                max_steps + 1,
                desc="Training model",
                leave=True,
                mininterval=5.0,
                total=max_steps,
                disable=not is_main_process,
                smoothing=0.3,
            ):
                if save_model and is_main_process and (step % checkpoint_interval) == 0:
                    nlp.to_disk(output_model_dir)
                if step > 0 and (step % validation_interval) == 0:
                    flat_cum_res = {
                        k: sum(v)
                        for k, v in ld_to_dl(gather_object([flat_cum_res])).items()
                    }
                    cum_res = decompress_dict(flat_cum_res)
                    if is_main_process:
                        cum_stats = decompress_dict(flat_cum_stats)
                        val_metrics = scorer(nlp, val_docs) if val_docs else {}
                        metrics = {
                            "step": step,
                            "lr": accel_optim.param_groups[0]["lr"],
                            "loss": cum_res["__all__"]["loss"] / count,
                            "count": count,
                            "spikes": spikes,
                            "weights": {
                                k: v / count for k, v in flat_cum_weight_metrics.items()
                            },
                            "per_pipe": {
                                n: {
                                    "stats": cum_stats[n].get("stats", {}),
                                    "results": (
                                        {
                                            k: v
                                            for k, v in trained_pipes_local[n]
                                            .compute_training_metrics(
                                                cum_res[n],
                                                cum_stats[n].get("stats", {}),
                                                count,
                                            )
                                            .items()
                                            if v is not None
                                        }
                                    ),
                                }
                                for n in pipe_names
                            },
                            "validation": val_metrics,
                        }
                        accelerator.log(metrics, step=step)
                    count = 0
                    spikes = 0
                    flat_cum_stats = defaultdict(float)
                    flat_cum_res = defaultdict(float)
                    flat_cum_weight_metrics = defaultdict(float)

                    if on_validation_callback:
                        on_validation_callback(metrics)

                if step == max_steps:
                    break

                accel_optim.zero_grad()

                batches = list(next(iterator))
                batches_pipe_names = list(
                    flatten_once(
                        [
                            [td.pipe_names or pipe_names] * len(b)
                            for td, b in zip(phase_training_data, batches)
                        ]
                    )
                )
                batches = list(flatten(batches))

                # Synchronize stats between sub-batches across workers
                local_batch_stats = {}
                for b in batches:
                    deep_add_flat(b, result=local_batch_stats)
                flat_batch_stats = ld_to_dl(gather_object([local_batch_stats]))
                flat_batch_stats = {k: sum(v) for k, v in flat_batch_stats.items()}

                # for training purposes (ie have the right denominator in the losses)
                for b in batches:
                    deep_set_unflat(b, flat_batch_stats)

                # for logging purposes
                for k, v in flat_batch_stats.items():
                    flat_cum_stats[k] += flat_batch_stats[k]

                for idx, (batch, batch_pipe_names) in enumerate(
                    zip(batches, batches_pipe_names)
                ):
                    cache_ctx = (
                        nlp.cache() if len(batch_pipe_names) > 1 else nullcontext()
                    )
                    no_sync_ctx = (
                        accelerator.no_sync(trained_pipes)
                        if idx < len(batches) - 1
                        else nullcontext()
                    )
                    try:
                        with cache_ctx, no_sync_ctx:
                            all_res, loss = trained_pipes(
                                batch,
                                enable=batch_pipe_names,
                            )
                            deep_add_flat(all_res, flat_cum_res, check_stats_key=False)
                            flat_cum_res["__all__/loss"] += loss.item()
                            del all_res
                            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                                # Trick to ensure all trained parameters participate in
                                # the loss otherwise torch (in parallel) isn't happy as
                                # it cannot reliably know if a param gradients haven't
                                # been collected yet, or won't participate in the step
                                # at all.
                                loss0 = sum(p.sum() * 0 for p in grad_params)
                                accelerator.backward(loss + loss0)
                    except torch.cuda.OutOfMemoryError:  # pragma: no cover
                        print(
                            "Out of memory error encountered when processing a "
                            "batch with the following statistics:"
                        )
                        print(local_batch_stats)
                        raise
                    except Exception:
                        print(
                            "An error occurred when processing a batch with these"
                            "dimensions"
                        )
                        print(local_batch_stats)
                        raise

                    del loss

                del flat_batch_stats
                accelerator.unscale_gradients()

                # Log gradients
                if log_weight_grads and is_main_process:
                    for pipe_name, pipe in trained_pipes_local.items():
                        for param_name, param in pipe.named_parameters():
                            if param.grad is not None:
                                flat_cum_weight_metrics[
                                    f"grad_norm/{pipe_name}/{param_name}"
                                ] += param.grad.norm().item()
                                flat_cum_weight_metrics[
                                    f"param_norm/{pipe_name}/{param_name}"
                                ] += param.norm().item()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    grad_params, grad_max_norm, norm_type=2
                ).item()

                # Detect grad spikes and skip the step if necessary
                if grad_dev_policy is not None:
                    if step > grad_ewm_window and (
                        grad_norm - grad_mean
                    ) > grad_max_dev * math.sqrt(grad_var):
                        spike = True
                        spikes += 1
                    else:
                        grad_mean, grad_var, ewm_state = ewm_moments(
                            grad_norm, grad_ewm_window, state=ewm_state
                        )
                        spike = False

                    if spike and grad_dev_policy == "clip_mean":
                        torch.nn.utils.clip_grad_norm_(
                            grad_params, grad_mean, norm_type=2
                        )
                    elif spike and grad_dev_policy == "clip_threshold":
                        torch.nn.utils.clip_grad_norm_(
                            grad_params,
                            grad_mean + math.sqrt(grad_var) * grad_max_dev,
                            norm_type=2,
                        )

                if grad_dev_policy != "skip" or not spike:
                    accel_optim.step()

                count += 1
                flat_cum_weight_metrics["grad_norm/__all__"] += grad_norm

            del iterator

    # Should we put this in a finally block?
    accelerator.end_training()

    return nlp
