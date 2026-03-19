import collections
import copy
import hashlib
import importlib.util
import json
import logging
import math
import multiprocessing as mp
import os
import random
import re
import statistics
import subprocess
import sys
import time
import uuid
import warnings
from datetime import datetime
from threading import Event, Lock, Thread
from typing import Dict, List, Literal, Optional, Tuple, Union

# Ensure SIGTERM triggers a cold shutdown in workers (otherwise termination hangs)
os.environ.setdefault("REMAP_SIGTERM", "SIGQUIT")

import joblib
import optuna
import optuna.visualization as vis
from configobj import ConfigObj
from confit import Cli, Config
from confit.utils.collections import split_path
from confit.utils.random import set_seed
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.samplers import TPESampler
from pydantic import BaseModel, confloat, conint
from ruamel.yaml import YAML

from edsnlp.training.trainer import registry

app = Cli(pretty_exceptions_show_locals=False)

# disable transformers lib warn logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logger = logging.getLogger(__name__)

DEFAULT_GPU_HOUR = 1.0
CHECKPOINT = "study.pkl"
DVC_QUEUE_LOCK = Lock()
DVC_SUBMIT_LOCK = Lock()
WorkerHandle = Union[mp.pool.Pool, List[mp.Process]]
WORKER_POOL: Optional[WorkerHandle] = None
METRICS_FILENAME = "metrics.json"
DEFAULT_TRAIN_SEED_PATH = "train.seed"


class HyperparameterConfig(BaseModel, extra="forbid"):
    """
    A configuration model for hyperparameters used in optimization or tuning processes.
    """

    type: str
    alias: Optional[str] = None
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    step: Optional[Union[float, int]] = None
    log: Optional[bool] = None
    choices: Optional[List[Union[str, float, int, bool]]] = None

    def to_dict(self) -> dict:
        """
        Convert the hyperparameter configuration to a dictionary.
        Excludes unset and default values to provide a minimal representation.

        Returns:
            dict: A dictionary representation of the hyperparameter configuration.
        """
        return self.model_dump(exclude_unset=True, exclude_defaults=True)


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def is_plotly_installed() -> bool:
    """
    Check if Plotly is installed. If not warn the user.
    Plotly is needed by optuna.visualization to produce tuning visual results.

    Returns:
        bool: True if Plotly is installed, False otherwise.
    """
    if importlib.util.find_spec("plotly") is None:
        logger.warning(
            "Warning, Plotly is not installed."
            "Please install it if you want tuning visual features."
        )
        return False
    return True


def _load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}.")
    with open(config_path, encoding="utf-8") as file:
        content = file.read()
    if str(config_path).endswith((".yaml", ".yml")):
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.representer.add_representer(
            type(None),
            lambda self, _: self.represent_scalar("tag:yaml.org,2002:null", "null"),
        )
        raw_config = yaml.load(content)
        config = Config.from_yaml_str(content)
        return config, raw_config, "yaml"
    raw_config = ConfigObj(content.splitlines(), encoding="utf-8")
    config = Config.from_str(content)
    return config, raw_config, "cfg"


def _detect_gpu_ids() -> List[int]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip() != "":
        return [int(x) for x in env.split(",") if x.strip().isdigit()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return list(range(len(result.stdout.strip().splitlines())))
    except FileNotFoundError:
        pass
    return [0]


def _parse_duration_seconds(value: Optional[Union[float, int, str]]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"Unsupported timeout type: {type(value)}")
    text = value.strip().lower()
    if not text:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    match = re.fullmatch(r"(\d+(\.\d+)?)([smhd])", text)
    if match:
        amount = float(match.group(1))
        unit = match.group(3)
        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        return amount * multipliers[unit]
    raise ValueError(
        "Invalid timeout format. Use seconds (e.g. '90'), or unit suffix "
        "like '30s', '10m', '1.5h', or '2d'."
    )


def _get_train_config(config: Dict) -> Dict:
    train_config = config.get("train")
    if train_config is None:
        raise ValueError("Missing 'train' section in tuning config.")
    return train_config


def _is_json_logger(logger_config) -> bool:
    if isinstance(logger_config, str):
        return logger_config == "json"
    if isinstance(logger_config, dict):
        return logger_config.get("@loggers") == "json"
    return False


def _resolve_metrics_relpath(config: Dict) -> str:
    train_config = _get_train_config(config)
    output_dir = train_config.get("output_dir") or "artifacts"
    logger_config = train_config.get("logger", True)

    if logger_config is True:
        file_name = METRICS_FILENAME
    elif not logger_config:
        raise ValueError(
            "DVC tuning requires the JSON logger. Set train.logger to include 'json'."
        )
    else:
        loggers = (
            list(logger_config)
            if isinstance(logger_config, (list, tuple))
            else [logger_config]
        )
        json_logger = next((item for item in loggers if _is_json_logger(item)), None)
        if json_logger is None:
            raise ValueError(
                "DVC tuning requires the JSON logger. "
                "Set train.logger to include 'json'."
            )
        file_name = (
            json_logger.get("file_name", METRICS_FILENAME)
            if isinstance(json_logger, dict)
            else METRICS_FILENAME
        )

    return os.path.join(str(output_dir), file_name)


def _compute_num_parallel_trials(
    gpu_ids: Optional[List[int]],
    training_seeds: Optional[List[int]],
    parallel_trials: bool,
) -> int:
    if not gpu_ids or not parallel_trials:
        return 1

    seeds_per_trial = len(training_seeds) if training_seeds else 1
    jobs = len(gpu_ids) // seeds_per_trial
    if jobs < 1:
        warnings.warn(
            "Not enough GPUs to run seeded trials in parallel; falling back to a "
            "single parallel Optuna job."
        )
        return 1
    return jobs


def _compute_importances(study, n=10):
    cumulative_importances = collections.defaultdict(float)

    for i in range(n):
        try:
            importance_scores = get_param_importances(
                study,
                evaluator=FanovaImportanceEvaluator(seed=i),
                target=lambda t: t.value,
            )
        except (RuntimeError, ValueError) as e:
            if "zero total variance" in str(e) or "only a single trial" in str(
                e
            ):  # pragma: no cover
                warnings.warn("Zero total variance : skipping importance computation.")
                continue
            raise
        for feature, importance in importance_scores.items():
            cumulative_importances[feature] += importance

    averaged_importances = {
        feature: total_importance / n
        for feature, total_importance in cumulative_importances.items()
    }

    sorted_importances = dict(
        sorted(averaged_importances.items(), key=lambda item: item[1], reverse=True)
    )
    return sorted_importances


def _resolve_choice_value(p_type: str, param_info: Dict, value):
    choices = param_info.get("choices")
    if (
        p_type == "ordered_categorical"
        and choices is not None
        and isinstance(value, int)
        and 0 <= value < len(choices)
    ):
        return choices[value]
    return value


def _suggest_param_value(
    trial: optuna.trial.Trial, name: str, p_type: str, suggest_dict: Dict
):
    if p_type == "int":
        return trial.suggest_int(name=name, **suggest_dict)
    if p_type == "float":
        return trial.suggest_float(name=name, **suggest_dict)
    if p_type == "ordered_categorical":
        choices = suggest_dict.pop("choices", None)
        if not choices:
            raise ValueError(
                f"Choices for ordered_categorical hyperparameter '{name}' "
                "cannot be empty."
            )
        value_index = trial.suggest_int(name=name, low=0, high=len(choices) - 1, step=1)
        return choices[value_index]
    if p_type == "categorical":
        return trial.suggest_categorical(name=name, **suggest_dict)
    raise ValueError(f"Unknown parameter type '{p_type}' for hyperparameter '{name}'.")


def update_config(
    config: Dict,
    tuned_parameters: Dict[str, Dict],
    values: Optional[Dict[str, any]] = None,
    trial: Optional[optuna.trial.Trial] = None,
    resolve: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Update a configuration dictionary with tuned hyperparameter values.

    This function modifies a given configuration dictionary by updating the specified
    hyperparameters with values from either a dictionary or an Optuna trial object.
    The updated configuration and training keyword arguments are returned.

    Parameters:
    -----------
    config : dict
        The configuration dictionary to be updated.
    tuned_parameters : dict
        A dictionary specifying the hyperparameters to tune.
    values : dict, optional
        A dictionary of parameter names and their corresponding values to update
        the configuration. Used when `trial` is not provided.
    trial : optuna.trial.Trial, optional
        An Optuna trial object to sample parameter values.
        Used when `values` is not provided.

    Returns:
    --------
    tuple
        - kwargs : dict
          The resolved training keyword arguments from the updated configuration.
        - updated_config : dict
          The modified configuration dictionary.
    """
    if values is None and trial is None:
        raise ValueError("Either 'values' or 'trial' parameters are expected.")

    for param, param_info in tuned_parameters.items():
        p_path = split_path(param)
        suggest_dict = dict(param_info)
        p_type = suggest_dict.pop("type")
        p_name = suggest_dict.pop("alias", None) or param

        if trial:
            value = _suggest_param_value(trial, p_name, p_type, suggest_dict)
        else:
            value = values.get(p_name, None)
            if value is None:
                continue
            value = _resolve_choice_value(p_type, param_info, value)

        current_config = config
        for key in p_path[:-1]:
            resolved_key = key
            if (
                isinstance(current_config, (tuple, list))
                and isinstance(key, str)
                and key.isdigit()
            ):
                resolved_key = int(key)
            try:
                current_config = current_config[resolved_key]
            except (KeyError, IndexError, TypeError) as exc:
                raise KeyError(f"Path '{key}' not found in config.") from exc
        current_config[p_path[-1]] = value

    if resolve:
        kwargs = Config.resolve(config["train"], registry=registry, root=config)
        return kwargs, config
    return config


def _build_xp_name(
    phase: int, trial_number: int, seed: int | str, study_name: str
) -> str:
    return f"{study_name}-p{phase}-t{trial_number:05d}-s{seed}"


def _build_worker_argv(node_name: str, loglevel: str) -> List[str]:
    return [
        "worker",
        f"--loglevel={loglevel}",
        f"--hostname={node_name}",
        "--pool=threads",
        "--concurrency=1",
        "--prefetch-multiplier=1",
        "--without-heartbeat",
        "--without-mingle",
        "--without-gossip",
    ]


def _build_overrides(
    tuned_parameters: Dict[str, Dict],
    values: Optional[Dict[str, any]] = None,
    trial: Optional[optuna.trial.Trial] = None,
) -> Dict[str, any]:
    if not values and not trial:
        raise ValueError("Either 'values' or 'trial' parameters are expected.")

    overrides: Dict[str, any] = {}
    for param, param_info in tuned_parameters.items():
        suggest_dict = dict(param_info)
        p_type = suggest_dict.pop("type")
        p_name = suggest_dict.pop("alias", None) or param
        if trial:
            value = _suggest_param_value(trial, p_name, p_type, suggest_dict)
        else:
            value = values.get(p_name, None)
            if value is None:
                continue
            value = _resolve_choice_value(p_type, param_info, value)
        overrides[f"{param}"] = value
    return overrides


def _worker_process(node_name: str, gpu_id: Optional[int], loglevel: str) -> None:
    from celery.utils.nodenames import default_nodename
    from dvc.repo import Repo

    if os.name == "nt":
        # see https://github.com/celery/billiard/issues/247
        os.environ["FORKED_BY_MULTIPROCESSING"] = "1"

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with Repo(".") as repo:
        queue = repo.experiments.celery_queue
        app = queue.celery

    if app.control.ping(
        destination=[default_nodename(node_name)],
        timeout=10.0,
    ):
        logger.info("Worker %s already running; skipping spawn.", node_name)
        return

    argv = _build_worker_argv(node_name, loglevel)
    app.worker_main(argv=argv)


# This is a bit brittle, since we try to mimic how DVC launches workers,
# while removing the empty queue monitoring logic.
def _start_worker_pool(gpu_ids: Optional[List[int]]) -> List[mp.Process]:
    ctx = mp.get_context("spawn")
    worker_gpu_ids = gpu_ids if gpu_ids else [None]
    from dvc.repo import Repo

    with Repo(".") as repo:
        queue = repo.experiments.celery_queue
        wdir_hash = hashlib.sha256(queue.wdir.encode("utf-8")).hexdigest()[:6]
    loglevel = "debug" if logger.getEffectiveLevel() <= logging.DEBUG else "info"
    processes: List[mp.Process] = []
    for num, gpu_id in enumerate(worker_gpu_ids, start=1):
        node_name = f"dvc-exp-{wdir_hash}-{num}@localhost"
        process = ctx.Process(
            target=_worker_process,
            args=(node_name, gpu_id, loglevel),
            daemon=True,
        )
        process.start()
        processes.append(process)
        logger.info("Started worker process %s on GPU %s", node_name, gpu_id)
    return processes


def _stop_worker_pool(pool: Optional[WorkerHandle]) -> None:
    if pool is None:
        return
    if isinstance(pool, list):
        for process in pool:
            if process.is_alive():
                process.terminate()
        for process in pool:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        return
    pool.terminate()
    pool.join()


def _enqueue_dvc_runs(
    *,
    repo,
    queue,
    training_seeds: Optional[List[int]],
    base_overrides: Dict[str, any],
    seed_path: str,
    config_path: str,
    phase: int,
    trial_number: int,
    study_name: str,
) -> Tuple[List[any], List[str]]:
    from dvc.utils.cli_parse import to_path_overrides

    entries = []
    names = []

    if training_seeds:
        DVC_SUBMIT_LOCK.acquire()
        try:
            for run_seed in training_seeds:
                xp_name = _build_xp_name(phase, trial_number, run_seed, study_name)
                overrides = dict(base_overrides)
                overrides[seed_path] = run_seed
                params = to_path_overrides(
                    [f"{config_path}:{k}={v}" for k, v in overrides.items()]
                )
                logger.info(f"Enqueuing experiment {xp_name}: {overrides}")
                entry = repo.experiments.queue_one(
                    queue,
                    params=params,
                    name=xp_name,
                )
                queue.wait_for_start(entry, sleep_interval=1)
                logger.info(f"Started {xp_name} experiment")
                entries.append(entry)
                names.append(xp_name)
        finally:
            DVC_SUBMIT_LOCK.release()
    else:
        xp_name = _build_xp_name(phase, trial_number, "base", study_name)
        params = to_path_overrides(
            [f"{config_path}:{key}={value}" for key, value in base_overrides.items()]
        )
        DVC_SUBMIT_LOCK.acquire()
        try:
            logger.info(f"Enqueuing experiment {xp_name}: {base_overrides}")
            entry = repo.experiments.queue_one(
                queue,
                params=params,
                name=xp_name,
            )
            queue.wait_for_start(entry, sleep_interval=1)
            logger.info(f"Started {xp_name} experiment")
        finally:
            DVC_SUBMIT_LOCK.release()
        entries.append(entry)
        names.append(xp_name)

    return entries, names


def _read_metric_from_rev(
    xp_name: str,
    metrics_relpath: str,
    metric_paths: List[Tuple[str, ...]],
) -> float:
    import dvc.api

    raw = dvc.api.read(metrics_relpath, repo=".", rev=xp_name)
    entries = json.loads(raw)
    if not isinstance(entries, list) or not entries:
        raise ValueError("metrics.json is empty or not a list")
    entry = entries[-1]
    score_dict = entry.get("validation")
    if score_dict is None:
        raise KeyError("Missing 'validation' in metrics entry")
    scores = []
    for path in metric_paths:
        score = score_dict
        for key in path:
            score = score[key]
        scores.append(score)
    if not all(isinstance(s, (int, float)) for s in scores):
        raise ValueError("Metric is not a scalar")
    return float(statistics.mean(scores))


def _extract_metric_from_validation(
    score_dict: Dict, metric_paths: List[Tuple[str, ...]]
) -> float:
    scores = []
    for path in metric_paths:
        score = score_dict
        for key in path:
            score = score[key]
        scores.append(score)
    if not all(isinstance(s, (int, float)) for s in scores):
        raise ValueError("Metric is not a scalar")
    return float(statistics.mean(scores))


def _load_metrics_entries(metrics_path: str) -> Optional[List[Dict]]:
    try:
        with open(metrics_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    return data


def _resolve_metrics_path(queue, entry, metrics_relpath: str) -> Optional[str]:
    from dvc.repo.experiments.executor.base import ExecutorInfo

    infofile = queue.get_infofile_path(entry.stash_rev)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if os.path.exists(infofile):
            try:
                info = ExecutorInfo.load_json(infofile)
            except Exception:
                time.sleep(1)
                continue
            return os.path.join(info.root_dir, metrics_relpath)
        time.sleep(1)
    return None


def _monitor_metrics_file(
    *,
    trial: optuna.trial.Trial,
    metric_paths: List[Tuple[str, ...]],
    metrics_path: str,
    stop_event: Event,
    pruned_event: Event,
    queue,
    entry,
    poll_interval: float = 1.0,
) -> None:
    last_len = 0
    while not stop_event.is_set():
        entries = _load_metrics_entries(metrics_path)
        if entries:
            if len(entries) < last_len:
                last_len = 0
            for idx in range(last_len, len(entries)):
                entry_metrics = entries[idx]
                try:
                    step = entry_metrics["step"]
                    score = _extract_metric_from_validation(
                        entry_metrics["validation"], metric_paths
                    )
                except Exception:
                    continue
                print("Reporting", score, "at", step)
                trial.report(score, step)
                # if trial.should_prune():
                #     pruned_event.set()
                #     try:
                #         queue.kill([entry.stash_rev])
                #     except Exception as exc:  # pragma: no cover - best effort
                #         logger.warning("Failed to kill pruned trial: %s", exc)
                #     stop_event.set()
                #     return
            last_len = len(entries)
        stop_event.wait(poll_interval)


def _collect_dvc_result(
    *,
    queue,
    entry,
    xp_name: str,
    trial: optuna.trial.Trial,
    metric_paths: List[Tuple[str, ...]],
    metrics_relpath: str,
) -> Tuple[Optional[any], bool]:
    stop_event = Event()
    pruned_event = Event()
    monitor = None
    try:
        queue.wait_for_start(entry, sleep_interval=1)
        metrics_path = _resolve_metrics_path(
            queue, entry, metrics_relpath=metrics_relpath
        )
        if metrics_path is None:
            raise RuntimeError(
                f"Unable to locate metrics file {metrics_relpath!r} "
                f"for experiment {xp_name}."
            )
        monitor = Thread(
            target=_monitor_metrics_file,
            kwargs={
                "trial": trial,
                "metric_paths": metric_paths,
                "metrics_path": metrics_path,
                "stop_event": stop_event,
                "pruned_event": pruned_event,
                "queue": queue,
                "entry": entry,
            },
            daemon=True,
        )
        monitor.start()
        result = queue.get_result(entry)
    except FileNotFoundError:
        result = None
    finally:
        stop_event.set()
        if monitor:
            monitor.join(timeout=5)

    return result, pruned_event.is_set()


def _handle_pruned_dvc_runs(queue, entries: List[any], current_entry) -> None:
    if len(entries) > 1:
        try:
            queue.kill([e.stash_rev for e in entries if e != current_entry])
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to kill remaining runs: %s", exc)
    raise optuna.TrialPruned()


def _objective_inprocess(
    config,
    tuned_parameters,
    trial,
    metric_paths: List[Tuple[str, ...]],
    phase: int,
    seed: int,
    training_seeds: Optional[List[int]],
):
    kwargs, _ = update_config(config, tuned_parameters, trial=trial)

    def extract_metric(score_dict: Dict) -> float:
        return _extract_metric_from_validation(score_dict, metric_paths)

    def on_validation_callback(all_metrics):
        step = all_metrics["step"]
        score = extract_metric(all_metrics["validation"])
        trial.report(score, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    def run_with_seed(run_seed: int) -> float:
        from edsnlp.training.trainer import GenericScorer, train

        set_seed(run_seed)
        nlp = train(**kwargs, on_validation_callback=on_validation_callback)
        scorer = GenericScorer(**kwargs["scorer"])
        val_data = kwargs["val_data"]
        score = scorer(nlp, val_data)
        return extract_metric(score)

    try:
        if training_seeds:
            scores = [run_with_seed(s) for s in training_seeds]
            scores = [s for s in scores if s is not None and s != float("-inf")]
            return sum(scores) / len(scores) if len(scores) > 0 else float("-inf")
        run_seed = (seed + trial.number) % (2**32)
        return run_with_seed(run_seed)
    except optuna.TrialPruned:
        logger.info("Trial pruned")
        raise


def _objective_dvc(
    config,
    tuned_parameters,
    trial,
    metric_paths: List[Tuple[str, ...]],
    phase: int,
    seed: int,
    seed_path: str,
    training_seeds: Optional[List[int]],
    config_path: str,
    study_name: str,
):
    from dvc.repo import Repo

    base_overrides = _build_overrides(tuned_parameters, trial=trial)
    metrics_relpath = _resolve_metrics_relpath(config)

    with Repo(".") as repo:
        queue = repo.experiments.celery_queue
        entries, names = _enqueue_dvc_runs(
            repo=repo,
            queue=queue,
            training_seeds=training_seeds,
            base_overrides=base_overrides,
            seed_path=seed_path,
            config_path=config_path,
            phase=phase,
            trial_number=trial.number,
            study_name=study_name,
        )

        scores = []
        for entry, xp_name in zip(entries, names):
            result, was_pruned = _collect_dvc_result(
                queue=queue,
                entry=entry,
                xp_name=xp_name,
                trial=trial,
                metric_paths=metric_paths,
                metrics_relpath=metrics_relpath,
            )
            if was_pruned:
                _handle_pruned_dvc_runs(queue, entries, entry)
            if not (result and result.exp_hash and result.ref_info):
                raise RuntimeError(f"dvc exp run failed for {xp_name}")
            score = _read_metric_from_rev(xp_name, metrics_relpath, metric_paths)
            scores.append(score)
            logger.info(f"Completed experiment {xp_name}: {score}")

    scores = [s for s in scores if s is not None and s != float("-inf")]
    return sum(scores) / len(scores) if len(scores) >= 1 else float("-inf")


def _objective_with_param(
    config,
    tuned_parameters,
    trial,
    metric_paths: List[Tuple[str, ...]],
    execution: Literal["inprocess", "dvc"] = "inprocess",
    phase: int = 1,
    seed: int = 42,
    seed_path: str = DEFAULT_TRAIN_SEED_PATH,
    training_seeds: Optional[List[int]] = None,
    config_path: str = "",
    study_name: str = "",
):
    if execution == "inprocess":
        return _objective_inprocess(
            config,
            tuned_parameters,
            trial,
            metric_paths,
            phase,
            seed,
            training_seeds,
        )
    return _objective_dvc(
        config,
        tuned_parameters,
        trial,
        metric_paths,
        phase,
        seed,
        seed_path,
        training_seeds,
        config_path=config_path,
        study_name=study_name,
    )


def _optimize(
    config_path,
    config_path_str: str,
    tuned_parameters,
    n_trials: Optional[int],
    metric_paths: List[str | Tuple[str, ...]],
    checkpoint_dir,
    study=None,
    *,
    execution: Literal["inprocess", "dvc"] = "inprocess",
    phase: int = 1,
    seed: int = 42,
    seed_path: str = DEFAULT_TRAIN_SEED_PATH,
    training_seeds: Optional[List[int]] = None,
    num_parallel_trials: int = 1,
    timeout: Optional[float] = None,
):
    if not study:
        # pruner = PercentilePruner(n_startup_trials=5, n_warmup_steps=2)
        study = optuna.create_study(
            direction="maximize",
            # pruner=pruner,
            sampler=TPESampler(seed=random.randint(0, 2**32 - 1)),
            study_name=f"optuna-{uuid.uuid4().hex[:8]}",
        )
    study_name = study.study_name

    def objective(trial):
        return _objective_with_param(
            config_path,
            tuned_parameters,
            trial,
            metric_paths,
            execution=execution,
            phase=phase,
            seed=seed,
            seed_path=seed_path,
            training_seeds=training_seeds,
            config_path=config_path_str,
            study_name=study_name,
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=num_parallel_trials,
        callbacks=[_save_checkpoint(checkpoint_dir)],
    )
    return study


def _save_checkpoint(checkpoint_dir):
    def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        checkpoint_file = os.path.join(checkpoint_dir, CHECKPOINT)
        logger.info(f"Saving checkpoint to {checkpoint_file}")
        joblib.dump(study, checkpoint_file)

    return callback


def _load_checkpoint(checkpoint_dir) -> Optional[optuna.study.Study]:
    checkpoint_file = os.path.join(checkpoint_dir, CHECKPOINT)
    if os.path.exists(checkpoint_file):
        logger.info(f"Loading study checkpoint from {checkpoint_file}")
        return joblib.load(checkpoint_file)
    return None


def _process_results(
    study,
    output_dir,
    viz,
    config_path,
    tuned_parameters,
    best_params_phase_1=None,
    raw_config=None,
    raw_kind: Optional[str] = None,
):
    importances = _compute_importances(study)

    # Some parameters of the study trials are stored as
    # ints but the actually map to choice values: we
    # map them back here.
    info_by_name = {
        (info.get("alias") or p): info for p, info in tuned_parameters.items()
    }
    best_params = {}
    for name, value in study.best_trial.params.items():
        info = info_by_name.get(name)
        if info:
            best_params[name] = _resolve_choice_value(info.get("type"), info, value)
        else:
            best_params[name] = value

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Value: {study.best_trial.value}")
    logger.info(f"Params: {best_params}")
    logger.info(f"Importances: {importances}")

    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "results_summary.txt")
    with open(results_file, "w") as f:
        f.write("Study Summary\n")
        f.write("==================\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"\nValue: {study.best_trial.value}\n")
        f.write("\nParams:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {json.dumps(value)}\n")
        if best_params_phase_1 is not None:
            for key_phase_1, value_phase_1 in best_params_phase_1.items():
                if key_phase_1 not in best_params:
                    f.write(f"  {key_phase_1}: {json.dumps(value_phase_1)}\n")
        f.write("\nImportances:\n")
        for key, value in importances.items():  # pragma: no cover
            f.write(f"  {key}: {json.dumps(value)}\n")

    _write_final_config(
        output_dir,
        config_path,
        tuned_parameters,
        best_params,
        raw_config=raw_config,
        raw_kind=raw_kind,
    )

    if viz:
        vis.plot_optimization_history(study).write_html(
            os.path.join(output_dir, "optimization_history.html")
        )
        vis.plot_parallel_coordinate(study).write_html(
            os.path.join(output_dir, "parallel_coordinate.html")
        )
        vis.plot_contour(study).write_html(os.path.join(output_dir, "contour.html"))
        vis.plot_edf(study).write_html(os.path.join(output_dir, "edf.html"))
        vis.plot_timeline(study).write_html(os.path.join(output_dir, "timeline.html"))

    return best_params, importances


def _write_final_config(
    output_dir,
    config_path,
    tuned_parameters,
    best_params,
    *,
    raw_config=None,
    raw_kind: Optional[str] = None,
):
    path_str = str(config_path)
    is_yaml = path_str.endswith(".yaml") or path_str.endswith(".yml")
    if raw_kind is None:
        raw_kind = "yaml" if is_yaml else "cfg"
    if raw_kind == "yaml":
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.representer.add_representer(
            type(None),
            lambda self, _: self.represent_scalar("tag:yaml.org,2002:null", "null"),
        )
        if raw_config is None:
            with open(config_path, encoding="utf-8") as file:
                original_config = yaml.load(file)
        else:
            original_config = copy.deepcopy(raw_config)
        updated_config = update_config(
            original_config, tuned_parameters, values=best_params, resolve=False
        )
        with open(
            os.path.join(output_dir, "config.yml"), "w", encoding="utf-8"
        ) as file:
            yaml.dump(updated_config, file)
    else:
        if raw_config is None:
            config = ConfigObj(config_path, encoding="utf-8")
        else:
            config = copy.deepcopy(raw_config)
        updated_config = update_config(
            dict(config), tuned_parameters, values=best_params, resolve=False
        )
        config.update(updated_config)
        config.filename = os.path.join(output_dir, "config.cfg")
        config.write()


def _parse_study_summary(output_dir):
    file_path = os.path.join(output_dir, "results_summary.txt")
    with open(file_path) as f:
        lines = f.readlines()

    sections = {"Params:": {}, "Importances:": {}}
    current = None

    for line in lines:
        line = line.strip()
        if line in sections:
            current = sections[line]
        elif current is not None and line:
            key, value = map(str.strip, line.split(":", 1))
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
            current[key] = parsed

    return sections["Params:"], sections["Importances:"]


def tune_two_phase(
    *,
    config: Dict,
    config_path: str,
    hyperparameters: Dict[str, Dict],
    output_dir: str,
    checkpoint_dir: str,
    n_trials: Optional[int],
    viz: bool,
    metric_paths: List[Tuple[str, ...]],
    study: Optional[optuna.study.Study] = None,
    skip_phase_1: bool = False,
    timeout: Optional[float] = None,
    execution: Literal["inprocess", "dvc"] = "inprocess",
    seed: int = 42,
    seed_path: str = DEFAULT_TRAIN_SEED_PATH,
    training_seeds: Optional[List[int]] = None,
    use_seeds_for_phase_1: bool = False,
    gpu_ids: Optional[List[int]] = None,
    parallel_trials: bool = False,
    raw_config=None,
    raw_kind: Optional[str] = None,
) -> None:
    """
    Perform two-phase hyperparameter tuning using Optuna.

    This method executes a two-phase tuning strategy. In the first phase, all specified
    hyperparameters are tuned. Based on their computed importance, only the most
    important hyperparameters (top 50% by importance) are selected for fine-tuning
    in the second phase, while the less important hyperparameters are frozen to their
    best values from phase one.

    Parameters:
    -----------
    config : dict
        The configuration dictionary for the model and training process.
    hyperparameters : dict
        A dictionary specifying the hyperparameters to tune.
    output_dir : str
        Directory where tuning results, visualizations, and best parameters will
        be saved.
    checkpoint_dir : str,
        Path to save the checkpoint file.
    n_trials : int, optional
        The total number of trials to execute across both tuning phases.
        This number will be split between the two phases, with approximately half
        of the trials assigned to each phase.
        If not provided, trials run until timeout for each phase.
    viz : bool
        Whether or not to include visual features (False if Plotly is unavailable).
    metric_paths : list[tuple[str, ...]]
        Metric paths used to evaluate trials (mean if multiple).
    study : optuna.study.Study, optional
        Optuna study containing previous trials when resuming from checkpoint.
    skip_phase_1 : bool, optional
        Whether or not to skip phase 1 (in case of resuming from checkpoint).
        Default is False.
    timeout : float, optional
        Timeout in seconds for each phase. If provided, it is applied to each phase.
    execution : {"inprocess", "dvc"}, optional
        Execution backend for trials. Default is "inprocess".
    seed : int, optional
        Base seed used for deterministic trial seeds. Default is 42.
    seed_path : str, optional
        Config path to the seed parameter for DVC overrides. Default is "train.seed".
    training_seeds : list[int], optional
        Fixed seed list for multi-seed aggregation. Default is None.
    use_seeds_for_phase_1 : bool, optional
        Use training_seeds during phase 1. Default is True for single-phase tuning,
        otherwise False.
    gpu_ids : list[int], optional
        List of GPU IDs to use for parallel trial execution. Default is None,
        which uses a single GPU.
    parallel_trials : bool, optional
        Whether to run trials in parallel across multiple GPUs when using DVC
        execution. Default is False.
    raw_config : object, optional
        Preloaded raw config to avoid re-reading config_path.
    raw_kind : {"yaml","cfg"}, optional
        Kind of raw_config when provided.
    """
    output_dir_phase_1 = os.path.join(output_dir, "phase_1")
    output_dir_phase_2 = os.path.join(output_dir, "phase_2")

    if str(config_path).endswith("yaml") or str(config_path).endswith("yml"):
        config_path_phase_2 = os.path.join(output_dir_phase_1, "config.yml")
    else:
        config_path_phase_2 = os.path.join(output_dir_phase_1, "config.cfg")

    if not skip_phase_1:
        if n_trials is None:
            n_trials_1 = None
            n_trials_2 = None
        else:
            n_trials_2 = n_trials // 2
            n_trials_1 = n_trials - n_trials_2

        if n_trials_1 is None:
            logger.info("Phase 1: Tuning all hyperparameters (timeout-based).")
        else:
            logger.info(f"Phase 1: Tuning all hyperparameters ({n_trials_1} trials).")

        study = _optimize(
            config,
            config_path,
            hyperparameters,
            n_trials_1,
            metric_paths,
            checkpoint_dir,
            study,
            execution=execution,
            phase=1,
            seed=seed,
            seed_path=seed_path,
            training_seeds=training_seeds if use_seeds_for_phase_1 else None,
            num_parallel_trials=_compute_num_parallel_trials(
                gpu_ids,
                training_seeds if use_seeds_for_phase_1 else None,
                parallel_trials,
            ),
            timeout=timeout,
        )
        best_params_phase_1, importances = _process_results(
            study,
            output_dir_phase_1,
            viz,
            config_path,
            hyperparameters,
            raw_config=raw_config,
            raw_kind=raw_kind,
        )

    else:
        n_trials_2 = n_trials
        logger.info("Skipping already tuned phase 1")
        best_params_phase_1, importances = _parse_study_summary(output_dir_phase_1)

    hyperparameters_to_keep = list(importances.keys())[
        : math.ceil(len(importances) / 2)
    ]

    hyperparameters_phase_2 = {
        key: value
        for key, value in hyperparameters.items()
        if key in hyperparameters_to_keep
        or (value.get("alias") and value["alias"] in hyperparameters_to_keep)
    }

    hyperparameters_frozen = {
        key: value
        for key, value in hyperparameters.items()
        if key not in hyperparameters_to_keep
        and (not value.get("alias") or value["alias"] not in hyperparameters_to_keep)
    }

    _, updated_config = update_config(
        config, hyperparameters_frozen, values=best_params_phase_1
    )

    if n_trials_2 is None:
        logger.info(
            "Phase 2: Tuning %s hyperparameters (timeout-based). "
            "Other hyperparameters frozen to best values.",
            hyperparameters_to_keep,
        )
    else:
        logger.info(
            "Phase 2: Tuning %s hyperparameters (%s trials). "
            "Other hyperparameters frozen to best values.",
            hyperparameters_to_keep,
            n_trials_2,
        )

    study = _optimize(
        updated_config,
        config_path_phase_2,
        hyperparameters_phase_2,
        n_trials_2,
        metric_paths,
        checkpoint_dir,
        study,
        execution=execution,
        phase=2,
        seed=seed,
        seed_path=seed_path,
        training_seeds=training_seeds,
        num_parallel_trials=_compute_num_parallel_trials(
            gpu_ids,
            training_seeds,
            parallel_trials,
        ),
        timeout=timeout,
    )

    _process_results(
        study,
        output_dir_phase_2,
        viz,
        config_path=config_path_phase_2,
        tuned_parameters=hyperparameters,
        best_params_phase_1=best_params_phase_1,
    )


@app.command(name="tuning", registry=registry)
def tune(
    *,
    config_meta: Dict,
    hyperparameters: Dict[str, HyperparameterConfig],
    output_dir: str,
    checkpoint_dir: str,
    gpu_hours: Optional[confloat(gt=0)] = None,
    timeout: Optional[Union[confloat(gt=0), str]] = None,
    n_trials: Optional[conint(gt=0)] = None,
    two_phase_tuning: bool = False,
    seed: int = 42,
    metric: Union[str, List[str]] = "ner.micro.f",
    keep_checkpoint: bool = False,
    execution: Literal["inprocess", "dvc"] = "inprocess",
    seed_path: str = DEFAULT_TRAIN_SEED_PATH,
    training_seeds: Optional[List[int]] = None,
    use_seeds_for_phase_1: Optional[bool] = None,
    gpu_ids: Optional[List[int]] = None,
    parallel_trials: bool = False,
):
    """
    Perform hyperparameter tuning for a model using Optuna.

    Parameters:
    -----------
    config_meta : dict
        Metadata for the configuration file, containing at least the key "config_path"
        which specifies the path to the configuration file.
    hyperparameters : dict
        A dictionary specifying the hyperparameters to tune. The keys are the parameter
        names, and the values are dictionaries containing the following fields:
        - "path": List[str] representing the path to the parameter in `config`.
        - "type": The type of parameter ("float", "int", "categorical",
          "ordered_categorical").
        - "low": (optional) Lower bound for numerical parameters.
        - "high": (optional) Upper bound for numerical parameters.
        - "step": (optional) Step size for numerical parameters.
        - "log": (optional) Whether to sample numerical parameters on a log scale.
        - "choices": (optional) List of values for categorical parameters.
          Required for "ordered_categorical".
    output_dir : str
        Directory where tuning results, visualizations, and best parameters will
        be saved.
    checkpoint_dir : str,
        Path to save the checkpoint file.
    gpu_hours : float, optional
        Deprecated. Total GPU time available for tuning, in hours. Used to
        derive `timeout` for backward compatibility.
    timeout : float or str, optional
        Total time budget for tuning, in seconds. You can also pass a duration
        string like "30s", "10m", "1.5h", or "2d". If `two_phase_tuning` is True,
        the timeout is split equally between the two phases.
    n_trials : int, optional
        Number of trials for tuning. If not provided, tuning runs until timeout.
    two_phase_tuning : bool, optional
        If True, performs two-phase tuning. In the first phase, all hyperparameters
        are tuned, and in the second phase, the top half (based on importance) are
        fine-tuned while freezing others.
        Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    metric : str or list[str], optional
        Metric(s) used to evaluate trials. If multiple, their mean is optimized.
        Default is "ner.micro.f".
    keep_checkpoint : bool, optional
        If True, keeps the checkpoint file after tuning. Default is False.
    execution : {"inprocess", "dvc"}, optional
        Execution backend for trials. Default is "inprocess".
    seed_path : str, optional
        Config path to the seed parameter for DVC overrides. Default is "train.seed".
    training_seeds : list[int], optional
        Fixed seed list for multi-seed aggregation. Default is None.
    use_seeds_for_phase_1 : bool, optional
        Use training_seeds during phase 1. Default is True for single-phase tuning,
        otherwise False.
    gpu_ids : list[int], optional
        GPU ids to use for DVC subprocesses. Default is None.
    parallel_trials : bool, optional
        Whether to run trials in parallel across multiple GPUs when using DVC, or
        run them sequentially (you may parametrize your training job to parallelize your
        batches across these GPUs). Default is False.

        !!! warning "

            When using `parallel_trials`, the 2nd, 3rd, etc. trials will be started
            at the same time, and will therefore not benefit from the previous trials'
            results. This may lead to suboptimal tuning results than
            running sequentially. However, if you have many GPUs and a time budget, and
            can only train on one GPU at a time, this is a good way to fully utilize
            your resources to improve your tuning performance (as you will get N times
            more trials in the same time).
    """
    _setup_logging()
    viz = is_plotly_installed()
    config_path = config_meta["config_path"][0]
    config, raw_config, raw_kind = _load_config(config_path)
    hyperparameters = {key: value.to_dict() for key, value in hyperparameters.items()}
    set_seed(seed)
    if isinstance(metric, (list, tuple)):
        metric_paths = [split_path(m) for m in metric]
    else:
        metric_paths = [split_path(metric)]
    if use_seeds_for_phase_1 is None:
        use_seeds_for_phase_1 = not two_phase_tuning
    study = _load_checkpoint(checkpoint_dir)
    elapsed_trials = 0
    skip_phase_1 = False
    if study:
        elapsed_trials = len(study.trials)
        logger.info(f"Elapsed trials: {elapsed_trials}")

    if execution == "dvc":
        lock_path = os.path.join(".dvc", "tmp", "lock")
        if os.path.exists(lock_path):
            raise RuntimeError(
                f"A DVC lock was found at {lock_path}, meaning an experiment is "
                "being launched at the moment, or a previous operation didn't "
                "finish properly. Remove it before you start the tunning."
            )
        global WORKER_POOL
        if parallel_trials:
            gpu_ids = gpu_ids or _detect_gpu_ids()
            WORKER_POOL = _start_worker_pool(gpu_ids)
        else:
            WORKER_POOL = _start_worker_pool(None)

    timeout_seconds = _parse_duration_seconds(timeout)
    gpu_hours = (
        DEFAULT_GPU_HOUR if n_trials is None and gpu_hours is None else gpu_hours
    )
    if timeout_seconds is None and gpu_hours is not None:
        timeout_seconds = gpu_hours * 3600

    try:
        if n_trials is not None and elapsed_trials >= (n_trials / 2):
            skip_phase_1 = True

        if n_trials is not None:
            n_trials = max(0, n_trials - elapsed_trials)
            logger.info(f"Number of trials: {n_trials}")
        else:
            logger.info("Number of trials: unlimited (timeout-based)")

        if two_phase_tuning:
            logger.info("Starting two-phase tuning.")
            phase_timeout = timeout_seconds / 2 if timeout_seconds is not None else None
            tune_two_phase(
                config=config,
                config_path=config_path,
                hyperparameters=hyperparameters,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                n_trials=n_trials,
                viz=viz,
                metric_paths=metric_paths,
                study=study,
                skip_phase_1=skip_phase_1,
                timeout=phase_timeout,
                execution=execution,
                seed=seed,
                seed_path=seed_path,
                training_seeds=training_seeds,
                use_seeds_for_phase_1=use_seeds_for_phase_1,
                gpu_ids=gpu_ids,
                parallel_trials=parallel_trials,
                raw_config=raw_config,
                raw_kind=raw_kind,
            )
        else:
            logger.info("Starting single-phase tuning.")
            study = _optimize(
                config,
                config_path,
                hyperparameters,
                n_trials,
                metric_paths,
                checkpoint_dir,
                study,
                execution=execution,
                phase=1,
                seed=seed,
                seed_path=seed_path,
                training_seeds=training_seeds if use_seeds_for_phase_1 else None,
                num_parallel_trials=_compute_num_parallel_trials(
                    gpu_ids,
                    training_seeds if use_seeds_for_phase_1 else None,
                    parallel_trials,
                ),
                timeout=timeout_seconds,
            )
            _process_results(study, output_dir, viz, config_path, hyperparameters)

    finally:
        if execution == "dvc":
            _stop_worker_pool(WORKER_POOL)
            WORKER_POOL = None

    logger.info(f"Tuning completed. Results available in {output_dir}.")
    checkpoint_file = os.path.join(checkpoint_dir, CHECKPOINT)
    if os.path.exists(checkpoint_file) and not keep_checkpoint:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archived = os.path.join(checkpoint_dir, f"study-{timestamp}.pkl")
        os.replace(checkpoint_file, archived)
        logger.info(
            "Archived checkpoint to %s. Rename it to %s and increase the number of "
            "trials to resume from this point.",
            archived,
            CHECKPOINT,
        )


if __name__ == "__main__":
    app()
