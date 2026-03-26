import os
import sys
import warnings
from types import SimpleNamespace
from unittest.mock import Mock, patch

import optuna
import pytest

from edsnlp.tune import (
    _build_overrides,
    _build_worker_argv,
    _build_xp_name,
    _collect_dvc_result,
    _compute_importances,
    _compute_num_parallel_trials,
    _detect_gpu_ids,
    _enqueue_dvc_runs,
    _handle_pruned_dvc_runs,
    _load_config,
    _load_metrics_entries,
    _monitor_metrics_file,
    _objective_dvc,
    _objective_inprocess,
    _parse_duration_seconds,
    _parse_study_summary,
    _process_results,
    _resolve_choice_value,
    _resolve_metrics_path,
    _resolve_metrics_relpath,
    _stop_worker_pool,
    _suggest_param_value,
    _worker_process,
    is_plotly_installed,
    tune,
)

pytestmark = pytest.mark.ml


def build_trial(number, value, params):
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
    trial.state = optuna.trial.TrialState.COMPLETE
    trial.system_attrs = {}
    trial.user_attrs = {}
    return trial


@pytest.fixture
def study():
    study = Mock(spec=optuna.study.Study)
    study.study_name = "mock_study"
    study._is_multi_objective.return_value = False
    study.trials = [
        build_trial(0, 0.9, {"param1": 0.15, "param2": 0.3}),
        build_trial(1, 0.75, {"param1": 0.05, "param2": 0.2}),
        build_trial(2, 0.99, {"param1": 0.3, "param2": 0.25}),
    ]
    study.get_trials.return_value = study.trials
    study.best_trial = study.trials[2]
    return study


def test_compute_importances(study):
    importance = _compute_importances(study)
    assert importance == {"param2": 0.5239814153755754, "param1": 0.4760185846244246}


@pytest.mark.parametrize(
    "config_path", ["tests/tuning/config.yml", "tests/tuning/config.cfg"]
)
def test_process_results(study, tmpdir, config_path):
    output_dir = tmpdir.mkdir("output")
    hyperparameters = {
        "train.param1": {
            "type": "int",
            "alias": "param1",
            "low": 2,
            "high": 8,
            "step": 2,
        },
    }

    best_params, importances = _process_results(
        study,
        output_dir,
        False,
        config_path,
        hyperparameters,
    )

    assert isinstance(best_params, dict)
    assert isinstance(importances, dict)

    results_file = os.path.join(output_dir, "results_summary.txt")
    assert os.path.exists(results_file)

    if config_path.endswith(("yml", "yaml")):
        config_file = os.path.join(output_dir, "config.yml")
    else:
        config_file = os.path.join(output_dir, "config.cfg")
    assert os.path.exists(config_file)

    with open(config_file, encoding="utf-8") as f:
        content = f.read()
    assert "# My use" in content


def test_resolve_metrics_relpath_defaults():
    config = {"train": {}}
    assert _resolve_metrics_relpath(config) == os.path.join("artifacts", "metrics.json")


def test_resolve_metrics_relpath_custom_json_logger():
    config = {
        "train": {
            "output_dir": "custom-artifacts",
            "logger": [
                "rich",
                {"@loggers": "json", "file_name": "scores.json"},
            ],
        }
    }
    assert _resolve_metrics_relpath(config) == os.path.join(
        "custom-artifacts", "scores.json"
    )


def test_resolve_metrics_relpath_requires_json_logger():
    config = {"train": {"logger": ["rich"]}}
    with pytest.raises(ValueError, match="JSON logger"):
        _resolve_metrics_relpath(config)


def test_resolve_metrics_relpath_requires_train_section():
    with pytest.raises(ValueError, match="Missing 'train' section"):
        _resolve_metrics_relpath({})


def test_compute_num_parallel_trials_clamps_to_one():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jobs = _compute_num_parallel_trials(
            gpu_ids=[0],
            training_seeds=[1, 2],
            parallel_trials=True,
        )

    assert jobs == 1
    assert any("falling back" in str(w.message) for w in caught)


def test_compute_num_parallel_trials_returns_available_jobs():
    assert (
        _compute_num_parallel_trials(
            gpu_ids=[0, 1, 2, 3],
            training_seeds=[1, 2],
            parallel_trials=True,
        )
        == 2
    )


@patch("importlib.util.find_spec")
def test_plotly(mock_importlib_util_find_spec):
    mock_importlib_util_find_spec.return_value = None
    assert not is_plotly_installed()


@pytest.mark.parametrize(
    ("filename", "content", "expected_kind", "expected_value", "expected_raw_value"),
    [
        (
            "config.yml",
            "# My useful comment\na:\n  aa: 1\nb: 2\nc: test\n",
            "yaml",
            1,
            1,
        ),
        (
            "config.cfg",
            "# My useful comment\n[a]\naa = 2\n",
            "cfg",
            2,
            "2",
        ),
    ],
)
def test_load_config(
    tmpdir, filename, content, expected_kind, expected_value, expected_raw_value
):
    config_dir = tmpdir.mkdir("configs")
    config_path = os.path.join(config_dir, filename)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)

    config, raw_config, raw_kind = _load_config(config_path)

    assert config["a"]["aa"] == expected_value
    assert raw_config["a"]["aa"] == expected_raw_value
    assert raw_kind == expected_kind

    with pytest.raises(FileNotFoundError):
        _load_config("wrong_path")


def test_detect_gpu_ids_from_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 7, nope")
    assert _detect_gpu_ids() == [2, 7]


def test_detect_gpu_ids_from_nvidia_smi(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with patch("subprocess.run") as run:
        run.return_value = SimpleNamespace(returncode=0, stdout="GPU 0\nGPU 1\n")
        assert _detect_gpu_ids() == [0, 1]


def test_detect_gpu_ids_fallback(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert _detect_gpu_ids() == [0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (90, 90.0),
        (1.5, 1.5),
        ("", None),
        ("90", 90.0),
        ("30s", 30.0),
        ("10m", 600.0),
        ("1.5h", 5400.0),
        ("2d", 172800.0),
    ],
)
def test_parse_duration_seconds(value, expected):
    assert _parse_duration_seconds(value) == expected


def test_parse_duration_seconds_rejects_invalid_values():
    with pytest.raises(ValueError, match="Unsupported timeout type"):
        _parse_duration_seconds([])
    with pytest.raises(ValueError, match="Invalid timeout format"):
        _parse_duration_seconds("later")


def test_compute_importances_skips_single_trial_variance(study):
    with patch(
        "edsnlp.tune.get_param_importances",
        side_effect=ValueError("only a single trial"),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _compute_importances(study) == {}
    assert any("skipping importance computation" in str(w.message) for w in caught)


def test_resolve_choice_value_and_build_overrides_from_values():
    assert (
        _resolve_choice_value("ordered_categorical", {"choices": ["a", "b", "c"]}, 1)
        == "b"
    )
    assert (
        _resolve_choice_value("ordered_categorical", {"choices": ["a", "b", "c"]}, 5)
        == 5
    )

    overrides = _build_overrides(
        {
            "train.batch_size": {
                "alias": "batch_size",
                "type": "ordered_categorical",
                "choices": [16, 32, 64],
            },
            "train.dropout": {
                "alias": "dropout",
                "type": "float",
                "low": 0.0,
                "high": 0.2,
            },
        },
        values={"batch_size": 2},
    )

    assert overrides == {"train.batch_size": 64}


def test_build_worker_argv():
    assert _build_worker_argv("worker@localhost", "info") == [
        "worker",
        "--loglevel=info",
        "--hostname=worker@localhost",
        "--pool=threads",
        "--concurrency=1",
        "--prefetch-multiplier=1",
        "--without-heartbeat",
        "--without-mingle",
        "--without-gossip",
    ]


def test_suggest_param_value_for_ordered_categorical():
    trial = Mock()
    trial.suggest_int.return_value = 1

    assert (
        _suggest_param_value(
            trial,
            "batch_size",
            "ordered_categorical",
            {"choices": [16, 32, 64]},
        )
        == 32
    )
    trial.suggest_int.assert_called_once_with(name="batch_size", low=0, high=2, step=1)

    with pytest.raises(ValueError, match="cannot be empty"):
        _suggest_param_value(
            trial, "batch_size", "ordered_categorical", {"choices": []}
        )


def test_load_metrics_entries_invalid_payloads(tmp_path):
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert _load_metrics_entries(str(invalid_json)) is None

    not_a_list = tmp_path / "metrics.json"
    not_a_list.write_text('{"step": 1}', encoding="utf-8")
    assert _load_metrics_entries(str(not_a_list)) is None


def test_resolve_metrics_path_retries_after_transient_load_error(monkeypatch):
    queue = Mock()
    queue.get_infofile_path.return_value = "info.json"
    monkeypatch.setattr("edsnlp.tune.time.sleep", lambda _: None)

    with patch("os.path.exists", return_value=True):
        with patch(
            "dvc.repo.experiments.executor.base.ExecutorInfo.load_json",
            side_effect=[
                RuntimeError("not ready"),
                SimpleNamespace(root_dir="/tmp/root"),
            ],
        ):
            assert (
                _resolve_metrics_path(
                    queue, SimpleNamespace(stash_rev="rev"), "metrics.json"
                )
                == "/tmp/root/metrics.json"
            )


def test_monitor_metrics_file_resets_and_skips_invalid_entries():
    class FakeStopEvent:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            return self.calls >= 2

        def wait(self, _):
            self.calls += 1

    trial = Mock()
    stop_event = FakeStopEvent()

    with patch(
        "edsnlp.tune._load_metrics_entries",
        side_effect=[
            [
                {"step": 1, "validation": {"ner": {"micro": {"f": 0.1}}}},
                {"step": 99, "validation": {"ner": {"micro": {"f": "bad"}}}},
            ],
            [{"step": 2, "validation": {"ner": {"micro": {"f": 0.2}}}}],
        ],
    ):
        _monitor_metrics_file(
            trial=trial,
            metric_paths=[("ner", "micro", "f")],
            metrics_path="unused.json",
            stop_event=stop_event,  # type: ignore
            pruned_event=Mock(),
            queue=Mock(),
            entry=Mock(),
            poll_interval=0,
        )

    assert trial.report.call_args_list == [((0.1, 1),), ((0.2, 2),)]


def test_stop_worker_pool_handles_none_list_and_pool():
    _stop_worker_pool(None)

    process = Mock()
    process.is_alive.side_effect = [True, True]
    _stop_worker_pool([process])
    process.terminate.assert_called_once()
    assert process.join.call_count == 2
    process.kill.assert_called_once()

    pool = Mock()
    _stop_worker_pool(pool)
    pool.terminate.assert_called_once()
    pool.join.assert_called_once()


def test_enqueue_dvc_runs_for_seeded_trials():
    repo = Mock()
    queue = Mock()
    repo.experiments.queue_one.side_effect = [
        SimpleNamespace(stash_rev="rev-1"),
        SimpleNamespace(stash_rev="rev-2"),
    ]

    with patch.dict(
        sys.modules,
        {
            "dvc.utils.cli_parse": SimpleNamespace(
                to_path_overrides=lambda params: params
            )
        },
    ):
        entries, names = _enqueue_dvc_runs(
            repo=repo,
            queue=queue,
            training_seeds=[11, 12],
            base_overrides={"train.lr": 0.1},
            seed_path="train.seed",
            config_path="config.yml",
            phase=2,
            trial_number=3,
            study_name="optuna-abcd",
        )

    assert [e.stash_rev for e in entries] == ["rev-1", "rev-2"]
    assert names == [
        _build_xp_name(2, 3, 11, "optuna-abcd"),
        _build_xp_name(2, 3, 12, "optuna-abcd"),
    ]
    assert repo.experiments.queue_one.call_count == 2
    queue.wait_for_start.assert_any_call(entries[0], sleep_interval=1)
    queue.wait_for_start.assert_any_call(entries[1], sleep_interval=1)


def test_enqueue_dvc_runs_for_single_trial():
    repo = Mock()
    queue = Mock()
    repo.experiments.queue_one.return_value = SimpleNamespace(stash_rev="rev-base")

    with patch.dict(
        sys.modules,
        {
            "dvc.utils.cli_parse": SimpleNamespace(
                to_path_overrides=lambda params: params
            )
        },
    ):
        entries, names = _enqueue_dvc_runs(
            repo=repo,
            queue=queue,
            training_seeds=None,
            base_overrides={"train.lr": 0.1},
            seed_path="train.seed",
            config_path="config.yml",
            phase=1,
            trial_number=5,
            study_name="optuna-abcd",
        )

    assert [e.stash_rev for e in entries] == ["rev-base"]
    assert names == [_build_xp_name(1, 5, "base", "optuna-abcd")]
    queue.wait_for_start.assert_called_once_with(entries[0], sleep_interval=1)


def test_collect_dvc_result_handles_missing_result_file():
    queue = Mock()
    entry = SimpleNamespace(stash_rev="rev-1")
    monitor = Mock()

    with patch("edsnlp.tune._resolve_metrics_path", return_value="/tmp/metrics.json"):
        with patch("edsnlp.tune.Thread", return_value=monitor):
            queue.get_result.side_effect = FileNotFoundError()
            result, was_pruned = _collect_dvc_result(
                queue=queue,
                entry=entry,
                xp_name="xp",
                trial=Mock(),
                metric_paths=[("ner", "micro", "f")],
                metrics_relpath="artifacts/metrics.json",
            )

    assert result is None
    assert was_pruned is False
    monitor.start.assert_called_once()
    monitor.join.assert_called_once_with(timeout=5)


def test_collect_dvc_result_reports_pruned_state():
    queue = Mock()
    entry = SimpleNamespace(stash_rev="rev-1")
    stop_event = Mock()
    stop_event.is_set.return_value = False
    pruned_event = Mock()
    pruned_event.is_set.return_value = True
    monitor = Mock()

    with patch("edsnlp.tune._resolve_metrics_path", return_value="/tmp/metrics.json"):
        with patch("edsnlp.tune.Thread", return_value=monitor):
            with patch("edsnlp.tune.Event", side_effect=[stop_event, pruned_event]):
                queue.get_result.return_value = SimpleNamespace(
                    exp_hash="hash",
                    ref_info=object(),
                )
                result, was_pruned = _collect_dvc_result(
                    queue=queue,
                    entry=entry,
                    xp_name="xp",
                    trial=Mock(),
                    metric_paths=[("ner", "micro", "f")],
                    metrics_relpath="artifacts/metrics.json",
                )

    assert result.exp_hash == "hash"
    assert was_pruned is True
    stop_event.set.assert_called_once()
    monitor.join.assert_called_once_with(timeout=5)


def test_handle_pruned_dvc_runs_kills_other_entries():
    queue = Mock()
    entries = [
        SimpleNamespace(stash_rev="rev-1"),
        SimpleNamespace(stash_rev="rev-2"),
        SimpleNamespace(stash_rev="rev-3"),
    ]

    with pytest.raises(optuna.TrialPruned):
        _handle_pruned_dvc_runs(queue, entries, entries[1])

    queue.kill.assert_called_once_with(["rev-1", "rev-3"])


def test_worker_process_skips_spawn_when_worker_is_already_running(monkeypatch):
    queue = SimpleNamespace(celery=Mock())
    queue.celery.control.ping.return_value = [object()]

    class FakeRepo:
        def __init__(self, _path):
            self.experiments = SimpleNamespace(celery_queue=queue)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with patch.dict(
        sys.modules,
        {
            "celery.utils.nodenames": SimpleNamespace(
                default_nodename=lambda name: f"node:{name}"
            ),
            "dvc.repo": SimpleNamespace(Repo=FakeRepo),
        },
    ):
        _worker_process("worker@localhost", gpu_id=2, loglevel="info")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    queue.celery.worker_main.assert_not_called()


def test_worker_process_starts_worker_main_and_sets_windows_flag(monkeypatch):
    queue = SimpleNamespace(celery=Mock())
    queue.celery.control.ping.return_value = []

    class FakeRepo:
        def __init__(self, _path):
            self.experiments = SimpleNamespace(celery_queue=queue)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.delenv("FORKED_BY_MULTIPROCESSING", raising=False)
    monkeypatch.setattr("edsnlp.tune.os.name", "nt", raising=False)
    with patch.dict(
        sys.modules,
        {
            "celery.utils.nodenames": SimpleNamespace(
                default_nodename=lambda name: f"node:{name}"
            ),
            "dvc.repo": SimpleNamespace(Repo=FakeRepo),
        },
    ):
        _worker_process("worker@localhost", gpu_id=None, loglevel="debug")

    assert os.environ["FORKED_BY_MULTIPROCESSING"] == "1"
    queue.celery.worker_main.assert_called_once_with(
        argv=_build_worker_argv("worker@localhost", "debug")
    )


def test_objective_dvc_averages_seeded_scores():
    queue = Mock()
    repo = SimpleNamespace(experiments=SimpleNamespace(celery_queue=queue))

    class FakeRepo:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return repo

        def __exit__(self, exc_type, exc, tb):
            return False

    trial = Mock()
    trial.number = 4

    entries = [SimpleNamespace(stash_rev="rev-1"), SimpleNamespace(stash_rev="rev-2")]
    names = ["xp-1", "xp-2"]

    with patch.dict(sys.modules, {"dvc.repo": SimpleNamespace(Repo=FakeRepo)}):
        with patch("edsnlp.tune._build_overrides", return_value={"train.lr": 0.1}):
            with patch(
                "edsnlp.tune._resolve_metrics_relpath",
                return_value="artifacts/metrics.json",
            ):
                with patch(
                    "edsnlp.tune._enqueue_dvc_runs",
                    return_value=(entries, names),
                ) as enqueue:
                    with patch(
                        "edsnlp.tune._collect_dvc_result",
                        side_effect=[
                            (SimpleNamespace(exp_hash="a", ref_info=object()), False),
                            (SimpleNamespace(exp_hash="b", ref_info=object()), False),
                        ],
                    ) as collect:
                        with patch(
                            "edsnlp.tune._read_metric_from_rev",
                            side_effect=[0.2, 0.8],
                        ):
                            value = _objective_dvc(
                                config={"train": {"logger": True}},
                                tuned_parameters={},
                                trial=trial,
                                metric_paths=[("ner", "micro", "f")],
                                phase=1,
                                seed=42,
                                seed_path="train.seed",
                                training_seeds=[1, 2],
                                config_path="config.yml",
                                study_name="optuna-abcd",
                            )

    assert value == pytest.approx(0.5)
    enqueue.assert_called_once()
    assert collect.call_count == 2


def test_objective_dvc_raises_when_run_fails():
    queue = Mock()
    repo = SimpleNamespace(experiments=SimpleNamespace(celery_queue=queue))

    class FakeRepo:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return repo

        def __exit__(self, exc_type, exc, tb):
            return False

    trial = Mock()
    trial.number = 1
    entries = [SimpleNamespace(stash_rev="rev-1")]
    names = ["xp-1"]

    with patch.dict(sys.modules, {"dvc.repo": SimpleNamespace(Repo=FakeRepo)}):
        with patch("edsnlp.tune._build_overrides", return_value={"train.lr": 0.1}):
            with patch(
                "edsnlp.tune._resolve_metrics_relpath",
                return_value="artifacts/metrics.json",
            ):
                with patch(
                    "edsnlp.tune._enqueue_dvc_runs",
                    return_value=(entries, names),
                ):
                    with patch(
                        "edsnlp.tune._collect_dvc_result",
                        return_value=(
                            SimpleNamespace(exp_hash=None, ref_info=None),
                            False,
                        ),
                    ):
                        with pytest.raises(RuntimeError, match="dvc exp run failed"):
                            _objective_dvc(
                                config={"train": {"logger": True}},
                                tuned_parameters={},
                                trial=trial,
                                metric_paths=[("ner", "micro", "f")],
                                phase=1,
                                seed=42,
                                seed_path="train.seed",
                                training_seeds=None,
                                config_path="config.yml",
                                study_name="optuna-abcd",
                            )


@patch("edsnlp.training.trainer.GenericScorer")
@patch("edsnlp.training.trainer.train")
@patch("edsnlp.tune.update_config")
def test_objective_inprocess_averages_training_seeds(
    mock_update_config,
    mock_train,
    mock_generic_scorer,
):
    mock_update_config.return_value = ({"scorer": {}, "val_data": object()}, {})

    scorer = Mock(side_effect=[{"score": 0.4}, {"score": 0.6}])
    mock_generic_scorer.return_value = scorer

    def fake_train(**kwargs):
        kwargs["on_validation_callback"]({"step": 1, "validation": {"score": 0.2}})
        return object()

    mock_train.side_effect = fake_train
    trial = Mock()
    trial.number = 3
    trial.should_prune.return_value = False

    value = _objective_inprocess(
        config={},
        tuned_parameters={},
        trial=trial,
        metric_paths=[("score",)],
        phase=1,
        seed=42,
        training_seeds=[11, 12],
    )

    assert value == pytest.approx(0.5)
    assert trial.report.call_count == 2


def test_parse_study_summary_parses_json_float_and_string(tmp_path):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "results_summary.txt").write_text(
        "\n".join(
            [
                "Study Summary",
                "Params:",
                '  a: "x"',
                "  b: 1.5",
                "  c: plain-text",
                "Importances:",
                "  train.lr: 0.75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    params, importances = _parse_study_summary(output_dir)

    assert params == {"a": "x", "b": 1.5, "c": "plain-text"}
    assert importances == {"train.lr": 0.75}


@patch("edsnlp.tune._process_results")
@patch("edsnlp.tune._optimize")
@patch("edsnlp.tune._load_checkpoint")
@patch("edsnlp.tune._load_config")
@patch("edsnlp.tune.is_plotly_installed")
@patch("edsnlp.tune._stop_worker_pool")
@patch("edsnlp.tune._start_worker_pool")
def test_tune_single_phase_parallel_jobs(
    mock_start_worker_pool,
    mock_stop_worker_pool,
    mock_is_plotly_installed,
    mock_load_config,
    mock_load_checkpoint,
    mock_optimize,
    mock_process_results,
    study,
):
    mock_is_plotly_installed.return_value = False
    mock_load_config.return_value = (
        {"train": {"logger": True}},
        {"train": {"logger": True}},
        "yaml",
    )
    mock_load_checkpoint.return_value = None
    mock_optimize.return_value = study
    mock_start_worker_pool.return_value = object()

    tune(
        config_meta={"config_path": ["fake_path"]},
        hyperparameters={},
        output_dir="output_dir",
        checkpoint_dir="checkpoint_dir",
        n_trials=4,
        seed=42,
        execution="dvc",
        parallel_trials=True,
        gpu_ids=[0],
        training_seeds=[1, 2],
    )

    assert mock_optimize.call_args.kwargs["num_parallel_trials"] == 1
    mock_process_results.assert_called_once()
    mock_start_worker_pool.assert_called_once_with([0])
    mock_stop_worker_pool.assert_called_once()


@patch("edsnlp.tune._process_results")
@patch("edsnlp.tune._optimize")
@patch("edsnlp.tune._load_checkpoint")
@patch("edsnlp.tune._load_config")
@patch("edsnlp.tune.is_plotly_installed")
def test_tune_parses_metric_list_and_timeout(
    mock_is_plotly_installed,
    mock_load_config,
    mock_load_checkpoint,
    mock_optimize,
    mock_process_results,
    study,
):
    mock_is_plotly_installed.return_value = False
    mock_load_config.return_value = (
        {"train": {"logger": True}},
        {"train": {"logger": True}},
        "yaml",
    )
    mock_load_checkpoint.return_value = None
    mock_optimize.return_value = study

    tune(
        config_meta={"config_path": ["fake_path"]},
        hyperparameters={},
        output_dir="output_dir",
        checkpoint_dir="checkpoint_dir",
        n_trials=2,
        metric=["ner.micro.f", "qual.micro.f"],
        timeout="1.5h",
    )

    assert mock_optimize.call_args.args[4] == [
        ("ner", "micro", "f"),
        ("qual", "micro", "f"),
    ]
    assert mock_optimize.call_args.kwargs["timeout"] == 5400.0
    mock_process_results.assert_called_once()


@patch("edsnlp.tune.tune_two_phase")
@patch("edsnlp.tune._load_checkpoint")
@patch("edsnlp.tune._load_config")
@patch("edsnlp.tune.is_plotly_installed")
def test_tune_two_phase_uses_keyword_call(
    mock_is_plotly_installed,
    mock_load_config,
    mock_load_checkpoint,
    mock_tune_two_phase,
):
    mock_is_plotly_installed.return_value = False
    mock_load_config.return_value = (
        {"train": {"logger": True}},
        {"train": {"logger": True}},
        "yaml",
    )
    mock_load_checkpoint.return_value = None

    tune(
        config_meta={"config_path": ["fake_path"]},
        hyperparameters={},
        output_dir="output_dir",
        checkpoint_dir="checkpoint_dir",
        n_trials=4,
        two_phase_tuning=True,
    )

    assert mock_tune_two_phase.called
    assert mock_tune_two_phase.call_args.kwargs["config_path"] == "fake_path"
