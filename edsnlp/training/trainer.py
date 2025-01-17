import json
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
    Collection,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Union,
)

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from confit import validate_arguments
from confit.utils.random import set_seed
from rich_logger import RichTablePrinter
from tqdm import tqdm, trange
from typing_extensions import Literal

from edsnlp import Pipeline, registry
from edsnlp.core.stream import Stream
from edsnlp.metrics.ner import NerMetric
from edsnlp.metrics.span_attributes import SpanAttributeMetric
from edsnlp.pipes.base import BaseNERComponent, BaseSpanAttributeClassifierComponent
from edsnlp.utils.batching import BatchSizeArg, stat_batchify
from edsnlp.utils.bindings import BINDING_SETTERS
from edsnlp.utils.collections import chain_zip, flatten, ld_to_dl
from edsnlp.utils.span_getters import get_spans
from edsnlp.utils.typing import AsList

from .optimizer import LinearSchedule, ScheduledOptimizer

LOGGER_FIELDS = {
    "step": {},
    "(.*)loss": {
        "goal": "lower_is_better",
        "format": "{:.2e}",
        "goal_wait": 2,
    },
    "lr": {"format": "{:.2e}"},
    "speed/(.*)": {"format": "{:.2f}", r"name": r"\1"},
    "labels": {"format": "{:.2f}"},
    "(.*?)/micro/(f|r|p)$": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 1,
        "name": r"\1_\2",
    },
    "(.*?)/(uas|las)": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 1,
        "name": r"\1_\2",
    },
}


def flatten_dict(d, path=""):
    if not isinstance(d, dict):
        return {path: d}

    return {
        k: v
        for key, val in d.items()
        for k, v in flatten_dict(val, f"{path}/{key}" if path else key).items()
    }


def fill_flat_stats(x, result, path=()):
    if result is None:
        result = {}
    if isinstance(x, dict):
        for k, v in x.items():
            fill_flat_stats(v, result, (*path, k))
        return result
    if "stats" in path and "__batch_hash__" not in path[-1]:
        path = "/".join(path)
        result[path] = result.get(path, 0) + x
    return result


def set_flat_stats(x, stats):
    for k, v in stats.items():
        path = k.split("/")
        current = x
        for p in path[:-1]:
            if p not in current:
                break
            current = current[p]
        else:
            current[path[-1]] = v


@validate_arguments
class GenericScorer:
    def __init__(
        self,
        speed: bool = True,
        batch_size: Union[int, str] = 1,
        autocast: Union[bool, Any] = None,
        **scorers,
    ):
        self.scorers = scorers
        self.speed = speed
        self.batch_size = batch_size
        self.autocast = autocast

    def __call__(self, nlp: Pipeline, docs: Iterable[Any]):
        scores = {}
        docs = list(docs)
        scorers = dict(self.scorers)

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
        ner_scorers = {
            name: scorers.pop(name)
            for name in list(scorers)
            if isinstance(scorers[name], NerMetric)
        }
        if ner_pipes and ner_scorers:
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
            for name, scorer in ner_scorers.items():
                scores[name] = scorer(docs, ner_preds)

        # Qualification
        qlf_pipes = [
            name
            for name, pipe in nlp.pipeline
            if isinstance(pipe, BaseSpanAttributeClassifierComponent)
        ]
        span_attr_scorers = {
            name: scorers.pop(name)
            for name in list(scorers)
            if isinstance(scorers[name], SpanAttributeMetric)
        }
        if qlf_pipes and span_attr_scorers:
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
            for name, scorer in span_attr_scorers.items():
                scores[name] = scorer(docs, qlf_preds)

        # Custom scorers
        for name, scorer in scorers.items():
            pred_docs = [d.copy() for d in tqdm(docs, desc="Copying docs")]
            preds = list(
                nlp.pipe(tqdm(pred_docs, desc="Predicting")).set_processing(
                    batch_size=self.batch_size,
                    autocast=self.autocast,
                )
            )
            scores[name] = scorer(docs, preds)

        return scores


if TYPE_CHECKING:
    GenericScorer = Union[GenericScorer, Dict]


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
        pipe_names: Optional[Collection[str]] = None,
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
        data = data.batchify(batch_size=self.batch_size[0], batch_by=batcher)
        if self.sub_batch_size:
            sub_batcher = stat_batchify(self.sub_batch_size[1] or "docs")
            data = data.map(
                lambda batch: [
                    nlp.collate(sub_batch, device=device)
                    for sub_batch in sub_batcher(batch, self.sub_batch_size[0])
                ]
            )
        else:
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
                    res[f"{name}_loss"] = res["loss"]
        return all_results, loss


@validate_arguments(registry=registry)
def train(
    *,
    nlp: Pipeline,
    train_data: AsList[TrainingData],
    val_data: AsList[Stream] = [],
    seed: int = 42,
    max_steps: int = 1000,
    optimizer: Union[ScheduledOptimizer, torch.optim.Optimizer] = None,
    validation_interval: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
    max_grad_norm: float = 5.0,
    loss_scales: Dict[str, float] = {},
    scorer: GenericScorer = GenericScorer(),
    num_workers: int = 0,
    cpu: bool = False,
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = "no",
    output_dir: Union[Path, str] = Path("artifacts"),
    output_model_dir: Optional[Union[Path, str]] = None,
    save_model: bool = True,
    logger: bool = True,
    config_meta: Optional[Dict] = None,
    on_validation_callback: Optional[Callable[[Dict], None]] = None,
    **kwargs,
):
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
    optimizer: Union[ScheduledOptimizer, torch.optim.Optimizer]
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
    max_grad_norm: float
        The maximum gradient norm
    loss_scales: Dict[str, float]
        The loss scales for each component (useful for multi-task learning)
    scorer: GenericScorer
        How to score the model. Expects a `GenericScorer` object or a dict
        containing a mapping of metric names to metric objects.
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
    logger: bool
        Whether to log the validation metrics in a rich table.
    on_validation_callback: Optional[Callable[[Dict], None]]
        A callback function invoked during validation steps to handle custom logic.
    kwargs: Dict
        Additional keyword arguments.

    Returns
    -------
    Pipeline
        The trained pipeline
    """
    # Prepare paths
    accelerator = Accelerator(cpu=cpu, mixed_precision=mixed_precision)
    # accelerator.register_for_checkpointing(dataset)
    is_main_process = accelerator.is_main_process
    device = accelerator.device

    output_dir = Path(output_dir or Path.cwd() / "artifacts")
    output_model_dir = output_model_dir or output_dir / "model-last"
    train_metrics_path = output_dir / "train_metrics.json"
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        if config_meta is not None:  # pragma: no cover
            print(config_meta["unresolved_config"].to_yaml_str())
            config_meta["unresolved_config"].to_disk(output_dir / "train_config.yml")

    validation_interval = validation_interval or max_steps // 10
    checkpoint_interval = checkpoint_interval or validation_interval
    trainable_pipe_names = {name for name, pipe in nlp.torch_components()}
    phases = nlp.connected_pipes_names()
    accelerator.print("Trainable components: " + ", ".join(trainable_pipe_names))
    accelerator.print(
        "Training phases:"
        + "".join(f"\n - {i + 1}: {', '.join(n)}" for i, n in enumerate(phases))
    )

    all_params = set(nlp.parameters())
    optim = optimizer
    del optimizer
    if optim is None:
        warnings.warn(
            "No optimizer provided, using default optimizer with default " "parameters"
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

    if kwargs:
        raise ValueError(f"Unknown arguments: {', '.join(kwargs)}")

    # Prepare validation docs
    val_docs = list(chain.from_iterable(val_data))

    # Initialize pipeline with training documents
    nlp.post_init(chain_zip([td.data for td in train_data if td.post_init]))

    for phase_i, pipe_names in enumerate(phases):
        trained_pipes = PipeDict({n: nlp.get_pipe(n) for n in pipe_names}, loss_scales)
        trained_pipes_params = set(trained_pipes.parameters())
        phase_training_data = [
            td
            for td in train_data
            if td.pipe_names is None or set(td.pipe_names) & set(pipe_names)
        ]

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

            accelerator.print(
                "Optimizing groups:"
                + "".join(
                    "\n - {} weight tensors ({:,} parameters){}".format(
                        len([p for p in g["params"] if p in grad_params]),
                        sum([p.numel() for p in g["params"] if p in grad_params]),
                        ": " + " & ".join(g.get("selectors", "*"))
                        if "selectors" in g
                        else "",
                    )
                    for g in optim.param_groups
                )
            )
            accelerator.print(
                f"Keeping frozen {len(all_params - grad_params):} weight tensors "
                f"({sum(p.numel() for p in all_params - grad_params):,} parameters)"
            )

            nlp.train(True)

            iterator = iter(
                zip(
                    *(
                        td(nlp, device).set_processing(
                            num_cpu_workers=num_workers,
                            process_start_method="spawn",
                        )
                        for td in phase_training_data
                    )
                )
            )
            (accel_optim, trained_pipes) = accelerator.prepare(optim, trained_pipes)
            if hasattr(accel_optim.optimizer, "initialize"):
                accel_optim.optimizer.initialize()

            cumulated_data = defaultdict(lambda: 0.0, count=0)
            all_metrics = []
            set_seed(seed)
            with (
                RichTablePrinter(LOGGER_FIELDS, auto_refresh=False)
                if is_main_process and logger
                else nullcontext()
            ) as logger:
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
                    if (
                        is_main_process
                        and step > 0
                        and (step % validation_interval) == 0
                    ):
                        scores = scorer(nlp, val_docs) if val_docs else {}
                        all_metrics.append(
                            {
                                "step": step,
                                "lr": accel_optim.param_groups[0]["lr"],
                                **cumulated_data,
                                **scores,
                            }
                        )
                        cumulated_data.clear()
                        train_metrics_path.write_text(json.dumps(all_metrics, indent=2))
                        if logger:
                            logger.log_metrics(flatten_dict(all_metrics[-1]))

                        if on_validation_callback:
                            on_validation_callback(all_metrics[-1])

                    if (
                        save_model
                        and is_main_process
                        and (step % checkpoint_interval) == 0
                    ):
                        nlp.to_disk(output_model_dir)

                    if step == max_steps:
                        break

                    accel_optim.zero_grad()

                    batches = list(next(iterator))
                    batches_pipe_names = list(
                        flatten(
                            [
                                [td.pipe_names or pipe_names] * len(b)
                                for td, b in zip(phase_training_data, batches)
                            ]
                        )
                    )
                    batches = list(flatten(batches))

                    # Synchronize stats between sub-batches across workers
                    batch_stats = {}
                    for b in batches:
                        fill_flat_stats(b, result=batch_stats)
                    batch_stats = {
                        k: sum(v)
                        for k, v in ld_to_dl(gather_object([batch_stats])).items()
                    }
                    for b in batches:
                        set_flat_stats(b, batch_stats)

                    res_stats = defaultdict(lambda: 0.0)
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
                        with cache_ctx, no_sync_ctx:
                            all_res, loss = trained_pipes(
                                batch,
                                enable=batch_pipe_names,
                            )
                            for name, res in all_res.items():
                                for k, v in res.items():
                                    if (
                                        isinstance(v, (float, int))
                                        or isinstance(v, torch.Tensor)
                                        and v.ndim == 0
                                    ):
                                        res_stats[k] += float(v)
                                    del k, v
                                del res
                            del all_res
                            accelerator.backward(loss)
                        del loss

                    # Sync output stats after forward such as losses, supports, etc.
                    res_stats = {
                        k: sum(v)
                        for k, v in ld_to_dl(gather_object([dict(res_stats)])).items()
                    }
                    if is_main_process:
                        for k, v in batch_stats.items():
                            cumulated_data[k] += v
                        for k, v in res_stats.items():
                            cumulated_data[k] += v

                    del batch_stats, res_stats
                    accelerator.clip_grad_norm_(grad_params, max_grad_norm)
                    accel_optim.step()

                del iterator

    return nlp
