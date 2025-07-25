import os

import pytest

try:
    import optuna
except ImportError:
    optuna = None

if optuna is None:
    pytest.skip("optuna not installed", allow_module_level=True)

import shutil

from edsnlp.tune import (
    tune,
)


def assert_results(output_dir):
    results_file = os.path.join(output_dir, "results_summary.txt")
    assert os.path.exists(results_file)

    with open(results_file, "r") as f:
        content = f.read()
        assert "Study Summary" in content
        assert "Best trial" in content
        assert "Value" in content
        assert "Params" in content
        assert "Importances" in content

    config_file = os.path.join(output_dir, "config.yml")
    assert os.path.exists(config_file), f"Expected file {config_file} not found"

    optimization_history_file = os.path.join(output_dir, "optimization_history.html")
    assert os.path.exists(optimization_history_file), (
        f"Expected file {optimization_history_file} not found"
    )

    parallel_coord_file = os.path.join(output_dir, "parallel_coordinate.html")
    assert os.path.exists(parallel_coord_file), (
        f"Expected file {parallel_coord_file} not found"
    )

    contour_file = os.path.join(output_dir, "contour.html")
    assert os.path.exists(contour_file), f"Expected file {contour_file} not found"

    edf_file = os.path.join(output_dir, "edf.html")
    assert os.path.exists(edf_file), f"Expected file {edf_file} not found"

    timeline_file = os.path.join(output_dir, "timeline.html")
    assert os.path.exists(timeline_file), f"Expected file {timeline_file} not found"


@pytest.mark.parametrize(
    "two_phase_tuning,n_trials,start_from_checkpoint",
    [
        (False, None, False),
        (False, 7, False),
        (False, 7, True),
        (True, None, False),
        (True, 7, False),
        (True, 7, True),
    ],
)
def test_tune(tmpdir, two_phase_tuning, n_trials, start_from_checkpoint):
    config_meta = {"config_path": ["tests/tuning/config.yml"]}
    hyperparameters = {
        "optimizer.groups.'.*'.lr.start_value": {
            "alias": "start_value",
            "type": "float",
            "low": 1e-4,
            "high": 1e-3,
            "log": True,
        },
        "optimizer.groups.'.*'.lr.warmup_rate": {
            "alias": "warmup_rate",
            "type": "float",
            "low": 0.0,
            "high": 0.3,
            "step": 0.05,
        },
    }
    output_dir = "./results"
    checkpoint_dir = "./tests/tuning/test_checkpoints"
    try:
        gpu_hours = 0.015 if n_trials is None else None
        seed = 42
        metric = "ner.micro.f"
        os.makedirs(checkpoint_dir, exist_ok=True)
        if start_from_checkpoint:
            assert gpu_hours is None
            test_schedule = [(n_trials, True), (n_trials, False)]
        else:
            test_schedule = [(n_trials, False)]
        for n_trials_, keep_checkpoint in test_schedule:
            tune(
                config_meta=config_meta,
                hyperparameters=hyperparameters,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                gpu_hours=gpu_hours,
                n_trials=n_trials_,
                two_phase_tuning=two_phase_tuning,
                seed=seed,
                metric=metric,
                keep_checkpoint=keep_checkpoint,
            )
        if two_phase_tuning:
            phase_1_dir = os.path.join(output_dir, "phase_1")
            phase_2_dir = os.path.join(output_dir, "phase_2")
            if not start_from_checkpoint:
                assert_results(phase_1_dir)
            assert_results(phase_2_dir)
        else:
            assert_results(output_dir)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree("./artifacts", ignore_errors=True)
