import datetime
import os
from unittest.mock import Mock, patch

import pytest

try:
    import optuna
except ImportError:
    optuna = None

if optuna is None:
    pytest.skip("optuna not installed", allow_module_level=True)

from confit import Config

from edsnlp.tune import (
    compute_importances,
    compute_n_trials,
    compute_remaining_n_trials_possible,
    compute_time_per_trial,
    is_plotly_install,
    load_config,
    optimize,
    process_results,
    tune,
)


def build_trial(number, value, params, datetime_start, datetime_complete):
    trial = Mock(spec=optuna.trial.FrozenTrial)
    trial.number = number
    trial.value = value
    trial.values = [value]
    trial.params = params
    trial.distributions = {
        "param1": optuna.distributions.FloatDistribution(
            high=0.3,
            log=False,
            low=0.0,
            step=0.05,
        ),
        "param2": optuna.distributions.FloatDistribution(
            high=0.3,
            log=False,
            low=0.0,
            step=0.05,
        ),
    }
    trial.datetime_start = datetime_start
    trial.datetime_complete = datetime_complete
    trial.state = optuna.trial.TrialState.COMPLETE
    trial.system_attrs = {}
    trial.user_attrs = {}
    return trial


@pytest.fixture
def study():
    study = Mock(spec=optuna.study.Study)
    study.study_name = "mock_study"
    study._is_multi_objective.return_value = False

    trials = []
    trial_0 = build_trial(
        number=0,
        value=0.9,
        params={"param1": 0.15, "param2": 0.3},
        datetime_start=datetime.datetime(2025, 1, 1, 12, 0, 0),
        datetime_complete=datetime.datetime(2025, 1, 1, 12, 5, 0),
    )
    trials.append(trial_0)

    trial_1 = build_trial(
        number=1,
        value=0.75,
        params={"param1": 0.05, "param2": 0.2},
        datetime_start=datetime.datetime(2025, 1, 1, 12, 5, 0),
        datetime_complete=datetime.datetime(2025, 1, 1, 12, 10, 0),
    )
    trials.append(trial_1)

    trial_2 = build_trial(
        number=2,
        value=0.99,
        params={"param1": 0.3, "param2": 0.25},
        datetime_start=datetime.datetime(2025, 1, 1, 12, 10, 0),
        datetime_complete=datetime.datetime(2025, 1, 1, 12, 15, 0),
    )
    trials.append(trial_2)

    study.trials = trials
    study.get_trials.return_value = trials
    study.best_trial = trials[2]
    return study


@pytest.mark.parametrize("ema", [True, False])
def test_compute_time_per_trial_with_ema(study, ema):
    result = compute_time_per_trial(study, ema=ema, alpha=0.1)
    assert result == pytest.approx(300.00)


@pytest.mark.parametrize(
    "gpu_hours, time_per_trial, expected_n_trials, raises_exception",
    [
        (1, 120, 30, False),
        (0.5, 3600, None, True),
    ],
)
def test_compute_n_trials(
    gpu_hours, time_per_trial, expected_n_trials, raises_exception
):
    if raises_exception:
        with pytest.raises(ValueError):
            compute_n_trials(gpu_hours, time_per_trial)
    else:
        result = compute_n_trials(gpu_hours, time_per_trial)
        assert result == expected_n_trials


def test_compute_importances(study):
    importance = compute_importances(study)
    assert importance == {"param2": 0.5239814153755754, "param1": 0.4760185846244246}


@pytest.mark.parametrize("viz", [True, False])
@pytest.mark.parametrize(
    "config_path", ["tests/tuning/config.yml", "tests/tuning/config.cfg"]
)
def test_process_results(study, tmpdir, viz, config_path):
    output_dir = tmpdir.mkdir("output")
    config = {
        "train": {
            "param1": None,
        },
        ".lr": {
            "param2": 0.01,
        },
    }
    hyperparameters = {
        "train.param1": {
            "type": "int",
            "alias": "param1",
            "low": 2,
            "high": 8,
            "step": 2,
        },
    }
    best_params, importances = process_results(
        study, output_dir, viz, config, config_path, hyperparameters
    )

    assert isinstance(best_params, dict)
    assert isinstance(importances, dict)

    results_file = os.path.join(output_dir, "results_summary.txt")
    assert os.path.exists(results_file)

    with open(results_file, "r") as f:
        content = f.read()
        assert "Study Summary" in content
        assert "Best trial" in content
        assert "Value" in content
        assert "Params" in content
        assert "Importances" in content

    if config_path.endswith("yml") or config_path.endswith("yaml"):
        config_file = os.path.join(output_dir, "config.yml")
    else:
        config_file = os.path.join(output_dir, "config.cfg")
    assert os.path.exists(config_file), f"Expected file {config_file} not found"

    with open(config_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert (
        "# My usefull comment" in content
    ), f"Expected comment not found in {config_file}"

    if viz:
        optimization_history_file = os.path.join(
            output_dir, "optimization_history.html"
        )
        assert os.path.exists(
            optimization_history_file
        ), f"Expected file {optimization_history_file} not found"

        parallel_coord_file = os.path.join(output_dir, "parallel_coordinate.html")
        assert os.path.exists(
            parallel_coord_file
        ), f"Expected file {parallel_coord_file} not found"

        contour_file = os.path.join(output_dir, "contour.html")
        assert os.path.exists(contour_file), f"Expected file {contour_file} not found"

        edf_file = os.path.join(output_dir, "edf.html")
        assert os.path.exists(edf_file), f"Expected file {edf_file} not found"

        timeline_file = os.path.join(output_dir, "timeline.html")
        assert os.path.exists(timeline_file), f"Expected file {timeline_file} not found"


def test_compute_remaining_n_trials_possible(study):
    gpu_hours = 0.5
    remaining_trials = compute_remaining_n_trials_possible(study, gpu_hours)
    assert remaining_trials == 3


@patch("edsnlp.tune.objective_with_param")
@patch("optuna.study.Study.optimize")
@pytest.mark.parametrize("has_study", [True, False])
def test_optimize(mock_objective_with_param, mock_optimize_study, has_study, study):
    mock_objective_with_param.return_value = 0.9
    metric = ("ner", "micro", "f")
    checkpoint_dir = "./checkpoint"

    if has_study:

        def pass_fn(obj, n_trials, callbacks):
            pass

        study.optimize = pass_fn
        study = optimize(
            "config_path",
            tuned_parameters={},
            n_trials=1,
            metric=metric,
            checkpoint_dir=checkpoint_dir,
            study=study,
        )
        assert isinstance(study, Mock)
        assert len(study.trials) == 3

    else:
        study = optimize(
            "config_path",
            tuned_parameters={},
            n_trials=1,
            metric=metric,
            checkpoint_dir=checkpoint_dir,
            study=None,
        )
        assert isinstance(study, optuna.study.Study)
        assert len(study.trials) == 0


@patch("edsnlp.tune.optimize")
@patch("edsnlp.tune.process_results")
@patch("edsnlp.tune.load_config")
@patch("edsnlp.tune.compute_n_trials")
@patch("edsnlp.tune.update_config")
@pytest.mark.parametrize("n_trials", [10, None])
@pytest.mark.parametrize("two_phase_tuning", [False, True])
def test_tune(
    mock_update_config,
    mock_compute_n_trials,
    mock_load_config,
    mock_process_results,
    mock_optimize,
    study,
    n_trials,
    two_phase_tuning,
):
    mock_load_config.return_value = {"train": {}, "scorer": {}, "val_data": {}}
    mock_update_config.return_value = None, {"train": {}, "scorer": {}, "val_data": {}}
    mock_optimize.return_value = study
    mock_process_results.return_value = ({}, {})
    mock_compute_n_trials.return_value = 10
    config_meta = {"config_path": ["fake_path"]}
    hyperparameters = {
        "param1": {"type": "float", "low": 0.0, "high": 1.0},
        "param2": {"type": "float", "low": 0.0, "high": 1.0},
    }
    output_dir = "output_dir"
    checkpoint_dir = "checkpoint_dir"
    gpu_hours = 0.25
    seed = 42

    tune(
        config_meta=config_meta,
        hyperparameters=hyperparameters,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        gpu_hours=gpu_hours,
        n_trials=n_trials,
        two_phase_tuning=two_phase_tuning,
        seed=seed,
    )

    mock_load_config.assert_called_once()

    if two_phase_tuning:
        if n_trials is None:
            assert mock_compute_n_trials.call_count == 2  # 1 at begining + 1 at end
            assert mock_optimize.call_count == 3  # 1 at begining + 2 for tuning
        else:
            mock_compute_n_trials.assert_not_called()
            assert mock_optimize.call_count == 2  # 2 for tuning

        assert mock_process_results.call_count == 2  # one for each phase

    else:
        if n_trials is None:
            assert mock_compute_n_trials.call_count == 2  # 1 at begining + 1 at end
            assert (
                mock_optimize.call_count == 3
            )  # 1 at begining + 1 for tuning + 1 at end
        else:
            mock_compute_n_trials.assert_not_called()
            assert mock_optimize.call_count == 1  # 1 for tuning

        mock_process_results.assert_called_once()


@patch("importlib.util.find_spec")
def test_plotly(mock_importlib_util_find_spec):
    mock_importlib_util_find_spec.return_value = None
    assert not is_plotly_install()


def test_load_config(tmpdir):
    cfg = """\
    "a":
        "aa": 1
    "b": 2
    "c": "test"
    """
    config_dir = tmpdir.mkdir("configs")
    config_path = os.path.join(config_dir, "config.yml")
    Config.from_yaml_str(cfg).to_disk(config_path)
    config = load_config(config_path)
    assert isinstance(config, Config)
    with pytest.raises(FileNotFoundError):
        load_config("wrong_path")
