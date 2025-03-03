import collections
import importlib.util
import logging
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple, Union

import joblib
import optuna
import optuna.visualization as vis
from configobj import ConfigObj
from confit import Cli, Config
from confit.utils.collections import split_path
from confit.utils.random import set_seed
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.pruners import MedianPruner
from pydantic import BaseModel, confloat, conint
from ruamel.yaml import YAML
from transformers.utils.logging import ERROR, set_verbosity

from edsnlp.training.trainer import GenericScorer, registry, train

app = Cli(pretty_exceptions_show_locals=False)

# disable transformers lib warn logs
set_verbosity(ERROR)

logger = logging.getLogger(__name__)

DEFAULT_GPU_HOUR = 1.0
CHECKPOINT = "study.pkl"


class HyperparameterConfig(BaseModel):
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

    class Config:
        extra = "forbid"

    def to_dict(self) -> dict:
        """
        Convert the hyperparameter configuration to a dictionary.
        Excludes unset and default values to provide a minimal representation.

        Returns:
            dict: A dictionary representation of the hyperparameter configuration.
        """
        return self.dict(exclude_unset=True, exclude_defaults=True)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def is_plotly_install() -> bool:
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


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}.")
    return Config.from_disk(config_path)


def compute_time_per_trial(
    study: optuna.study.Study, ema: bool = False, alpha: float = 0.1
) -> float:
    """
    Compute the time for the first trial or the EMA (Exponential Moving Average)
    across all trials in the study.

    Parameters:
    -----------
    study : optuna.study.Study
        An Optuna study object containing past trials.
    ema : bool
        If True, computes the EMA of trial times; otherwise, computes the
        time of the first trial.
    alpha : float, optional
        Smoothing factor for EMA. Only used if ema is True.

    Returns:
    --------
        float: Time for the first trial or the EMA across all trials, in seconds.
    """
    if ema:
        for i, trial in enumerate(study.trials):
            time_delta = (
                trial.datetime_complete - trial.datetime_start
            ).total_seconds()
            if i == 0:
                ema_time = time_delta
            else:
                ema_time = alpha * time_delta + (1 - alpha) * ema_time
        return ema_time
    else:
        trial = study.trials[0]
        time_delta = trial.datetime_complete - trial.datetime_start
        return time_delta.total_seconds()


def compute_n_trials(gpu_hours: float, time_per_trial: float) -> int:
    """
    Estimate the maximum number of trials that can be executed within a given
    GPU time budget.

    Parameters:
    -----------
    gpu_hours : float
        The total amount of GPU time available for tuning, in hours.
    time_per_trial : float
        Time per trial, in seconds.

    Returns:
    --------
        int: The number of trials that can be run within the given GPU time budget.
    """
    total_time_available = gpu_hours * 3600
    n_trials = int(total_time_available / time_per_trial)
    if n_trials <= 0:
        raise ValueError(
            "Not enough GPU time to tune hyperparameters."
            "Either raise your GPU time or specify more trials."
        )
    return n_trials


def compute_importances(study, n=10):
    cumulative_importances = collections.defaultdict(float)

    for i in range(n):
        importance_scores = get_param_importances(
            study,
            evaluator=FanovaImportanceEvaluator(seed=i),
            target=lambda t: t.value,
        )
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
    if not values and not trial:
        raise ValueError("Either 'values' or 'trial' parameters are expected.")

    for param, param_info in tuned_parameters.items():
        p_path = split_path(param)
        p_alias = param_info.get("alias", None)
        if p_alias:
            p_name = p_alias
        else:
            p_name = param

        if trial:
            p_type = param_info["type"]
            if p_type in ["float", "int"]:
                p_low = param_info["low"]
                p_high = param_info["high"]
                p_step = param_info.get("step", None)
                p_log = param_info.get("log", False)
                if p_type == "float":
                    value = trial.suggest_float(
                        name=p_name, low=p_low, high=p_high, step=p_step, log=p_log
                    )
                else:
                    value = trial.suggest_int(
                        name=p_name, low=p_low, high=p_high, step=p_step, log=p_log
                    )
            elif p_type == "categorical":
                p_choices = param_info["choices"]
                value = trial.suggest_categorical(name=p_name, choices=p_choices)
            else:
                raise ValueError(
                    f"Unknown parameter type '{p_type}' for hyperparameter '{p_name}'."
                )
        else:
            value = values.get(p_name, None)
            if value is None:
                continue

        current_config = config
        for key in p_path[:-1]:
            if key not in current_config:
                raise KeyError(f"Path '{key}' not found in config.")
            current_config = current_config[key]
        current_config[p_path[-1]] = value

    if resolve:
        kwargs = Config.resolve(config["train"], registry=registry, root=config)
        return kwargs, config
    return config


def objective_with_param(config, tuned_parameters, trial, metric):
    kwargs, _ = update_config(config, tuned_parameters, trial=trial)
    seed = random.randint(0, 2**32 - 1)
    set_seed(seed)

    def on_validation_callback(all_metrics):
        step = all_metrics["step"]
        score = all_metrics
        for key in metric:
            score = score[key]
        trial.report(score, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    try:
        nlp = train(**kwargs, on_validation_callback=on_validation_callback)
    except optuna.TrialPruned:
        logger.info("Trial pruned")
        raise
    scorer = GenericScorer(**kwargs["scorer"])
    val_data = kwargs["val_data"]
    score = scorer(nlp, val_data)
    for key in metric:
        score = score[key]
    return score


def optimize(
    config_path, tuned_parameters, n_trials, metric, checkpoint_dir, study=None
):
    def objective(trial):
        return objective_with_param(config_path, tuned_parameters, trial, metric)

    if not study:
        study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        )
    study.optimize(
        objective, n_trials=n_trials, callbacks=[save_checkpoint(checkpoint_dir)]
    )
    return study


def save_checkpoint(checkpoint_dir):
    def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        checkpoint_file = os.path.join(checkpoint_dir, CHECKPOINT)
        logger.info(f"Saving checkpoint to {checkpoint_file}")
        joblib.dump(study, checkpoint_file)

    return callback


def load_checkpoint(checkpoint_dir) -> Optional[optuna.study.Study]:
    checkpoint_file = os.path.join(checkpoint_dir, CHECKPOINT)
    if os.path.exists(checkpoint_file):
        logger.info(f"Loading study checkpoint from {checkpoint_file}")
        return joblib.load(checkpoint_file)
    return None


def process_results(
    study,
    output_dir,
    viz,
    config,
    config_path,
    tuned_parameters,
    best_params_phase_1=None,
):
    importances = compute_importances(study)
    best_params = study.best_trial.params

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
            f.write(f"  {key}: {value}\n")
        if best_params_phase_1 is not None:
            for key_phase_1, value_phase_1 in best_params_phase_1.items():
                if key_phase_1 not in best_params.keys():
                    f.write(f"  {key_phase_1}: {value_phase_1}\n")
        f.write("\nImportances:\n")
        for key, value in importances.items():
            f.write(f"  {key}: {value}\n")

    write_final_config(output_dir, config_path, tuned_parameters, best_params)

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


def write_final_config(output_dir, config_path, tuned_parameters, best_params):
    path_str = str(config_path)
    if path_str.endswith(".yaml") or path_str.endswith(".yml"):
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.representer.add_representer(
            type(None),
            lambda self, _: self.represent_scalar("tag:yaml.org,2002:null", "null"),
        )
        with open(config_path, "r", encoding="utf-8") as file:
            original_config = yaml.load(file)
        updated_config = update_config(
            original_config, tuned_parameters, values=best_params, resolve=False
        )
        with open(
            os.path.join(output_dir, "config.yml"), "w", encoding="utf-8"
        ) as file:
            yaml.dump(updated_config, file)
    else:
        config = ConfigObj(config_path, encoding="utf-8")
        updated_config = update_config(
            dict(config), tuned_parameters, values=best_params, resolve=False
        )
        config.update(updated_config)
        config.filename = os.path.join(output_dir, "config.cfg")
        config.write()


def parse_study_summary(output_dir):
    file_path = os.path.join(output_dir, "results_summary.txt")
    with open(file_path, "r") as f:
        lines = f.readlines()

    sections = {"Params:": {}, "Importances:": {}}
    current = None

    for line in lines:
        line = line.strip()
        if line in sections:
            current = sections[line]
        elif current is not None and line:
            key, value = map(str.strip, line.split(":"))
            current[key] = float(value)

    return sections["Params:"], sections["Importances:"]


def tune_two_phase(
    config: Dict,
    config_path: str,
    hyperparameters: Dict[str, Dict],
    output_dir: str,
    checkpoint_dir: str,
    n_trials: int,
    viz: bool,
    metric: Tuple[str],
    study: Optional[optuna.study.Study] = None,
    is_fixed_n_trials: bool = False,
    gpu_hours: float = 1.0,
    skip_phase_1: bool = False,
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
    n_trials : int
        The total number of trials to execute across both tuning phases.
        This number will be split between the two phases, with approximately half
        of the trials assigned to each phase.
    viz : bool
        Whether or not to include visual features (False if Plotly is unavailable).
    metric : Tuple[str]
        Metric used to evaluate trials.
    study : optuna.study.Study, optional
        Optuna study containing the first trial that was used to compute `n_trials`
        in case the user specifies a GPU hour budget.
    is_fixed_trial : bool, optional
        Whether or not the user specified fixed `n_trials` in config.
        If not, recompute n_trials between the two phases. In case there was multiples
        trials pruned in phase 1, we raise n_trials to compensate. Default is False.
    gpu_hours : float, optional
        Total GPU time available for tuning, in hours. Default is 1 hour.
    skip_phase_1 : bool, optional
        Whether or not to skip phase 1 (in case of resuming from checkpoint).
        Default is False.
    """
    output_dir_phase_1 = os.path.join(output_dir, "phase_1")
    output_dir_phase_2 = os.path.join(output_dir, "phase_2")

    if str(config_path).endswith("yaml") or str(config_path).endswith("yml"):
        config_path_phase_2 = os.path.join(output_dir_phase_1, "config.yml")
    else:
        config_path_phase_2 = os.path.join(output_dir_phase_1, "config.cfg")

    if not skip_phase_1:
        n_trials_2 = n_trials // 2
        n_trials_1 = n_trials - n_trials_2

        logger.info(f"Phase 1: Tuning all hyperparameters ({n_trials_1} trials).")
        study = optimize(
            config,
            hyperparameters,
            n_trials_1,
            metric,
            checkpoint_dir,
            study,
        )
        best_params_phase_1, importances = process_results(
            study, output_dir_phase_1, viz, config, config_path, hyperparameters
        )
        if not is_fixed_n_trials:
            n_trials_2 = compute_remaining_n_trials_possible(study, gpu_hours)

    else:
        n_trials_2 = n_trials
        logger.info("Skipping already tuned phase 1")
        best_params_phase_1, importances = parse_study_summary(output_dir_phase_1)

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

    logger.info(
        f"Phase 2: Tuning {hyperparameters_to_keep} hyperparameters "
        f"({n_trials_2} trials). Other hyperparameters frozen to best values."
    )

    study = optimize(
        updated_config,
        hyperparameters_phase_2,
        n_trials_2,
        metric,
        checkpoint_dir,
        study,
    )

    process_results(
        study,
        output_dir_phase_2,
        viz,
        config,
        config_path=config_path_phase_2,
        tuned_parameters=hyperparameters,
        best_params_phase_1=best_params_phase_1,
    )


def compute_remaining_n_trials_possible(
    study: optuna.study.Study,
    gpu_hours: float,
) -> int:
    """
    Compute the remaining number of trials possible within the GPU time budget
    that was not used by the study (in cases where multiple trials were pruned).

    Parameters:
    -----------
    study : optuna.study.Study
        An Optuna study object containing past trials.
    gpu_hours : float
        The total amount of GPU time available for tuning, in hours.

    Returns:
    --------
    int: The remaining number of trials possible.
    """
    first_trial = study.trials[0]
    last_trial = study.trials[-1]
    elapsed_gpu_time = (
        last_trial.datetime_complete - first_trial.datetime_start
    ).total_seconds()
    remaining_gpu_time = (gpu_hours * 3600 - elapsed_gpu_time) / 3600
    try:
        n_trials = compute_n_trials(
            remaining_gpu_time, compute_time_per_trial(study, ema=True)
        )
        return n_trials
    except ValueError:
        return 0


@app.command(name="tuning", registry=registry)
def tune(
    *,
    config_meta: Dict,
    hyperparameters: Dict[str, HyperparameterConfig],
    output_dir: str,
    checkpoint_dir: str,
    gpu_hours: confloat(gt=0) = DEFAULT_GPU_HOUR,
    n_trials: conint(gt=0) = None,
    two_phase_tuning: bool = False,
    seed: int = 42,
    metric="ner.micro.f",
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
        - "type": The type of parameter ("float", "int", "categorical").
        - "low": (optional) Lower bound for numerical parameters.
        - "high": (optional) Upper bound for numerical parameters.
        - "step": (optional) Step size for numerical parameters.
        - "log": (optional) Whether to sample numerical parameters on a log scale.
        - "choices": (optional) List of values for categorical parameters.
    output_dir : str
        Directory where tuning results, visualizations, and best parameters will
        be saved.
    checkpoint_dir : str,
        Path to save the checkpoint file.
    gpu_hours : float, optional
        Total GPU time available for tuning, in hours. Default is 1 hour.
    n_trials : int, optional
        Number of trials for tuning. If not provided, it will be computed based on the
        `gpu_hours` and the estimated time per trial.
    two_phase_tuning : bool, optional
        If True, performs two-phase tuning. In the first phase, all hyperparameters
        are tuned, and in the second phase, the top half (based on importance) are
        fine-tuned while freezing others.
        Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    """
    setup_logging()
    viz = is_plotly_install()
    config_path = config_meta["config_path"][0]
    config = load_config(config_path)
    hyperparameters = {key: value.to_dict() for key, value in hyperparameters.items()}
    set_seed(seed)
    metric = split_path(metric)
    study = load_checkpoint(checkpoint_dir)
    elapsed_trials = 0
    skip_phase_1 = False
    is_fixed_n_trials = n_trials is not None

    if study:
        elapsed_trials = len(study.trials)
        logger.info(f"Elapsed trials: {elapsed_trials}")

    if not is_fixed_n_trials:
        if not study:
            logger.info(f"Computing number of trials for {gpu_hours} hours of GPU.")
            study = optimize(
                config,
                hyperparameters,
                n_trials=1,
                metric=metric,
                checkpoint_dir=checkpoint_dir,
            )
            n_trials = compute_n_trials(gpu_hours, compute_time_per_trial(study)) - 1
        else:
            n_trials = compute_n_trials(
                gpu_hours, compute_time_per_trial(study, ema=True)
            )

    if elapsed_trials >= (n_trials / 2):
        skip_phase_1 = True

    n_trials = max(0, n_trials - elapsed_trials)

    logger.info(f"Number of trials: {n_trials}")

    if two_phase_tuning:
        logger.info("Starting two-phase tuning.")
        tune_two_phase(
            config,
            config_path,
            hyperparameters,
            output_dir,
            checkpoint_dir,
            n_trials,
            viz,
            metric=metric,
            study=study,
            is_fixed_n_trials=is_fixed_n_trials,
            gpu_hours=gpu_hours,
            skip_phase_1=skip_phase_1,
        )
    else:
        logger.info("Starting single-phase tuning.")
        study = optimize(
            config,
            hyperparameters,
            n_trials,
            metric,
            checkpoint_dir,
            study,
        )
        if not is_fixed_n_trials:
            n_trials = compute_remaining_n_trials_possible(study, gpu_hours)
            if n_trials > 0:
                logger.info(
                    f"As some trials were pruned, perform tuning for {n_trials} "
                    "more trials to fully use GPU time budget."
                )
                study = optimize(
                    config,
                    hyperparameters,
                    n_trials,
                    metric,
                    checkpoint_dir,
                    study,
                )
        process_results(study, output_dir, viz, config, config_path, hyperparameters)

    logger.info(
        f"Tuning completed. Results available in {output_dir}. Deleting checkpoint."
    )
    checkpoint_file = os.path.join(checkpoint_dir, CHECKPOINT)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


if __name__ == "__main__":
    app()
