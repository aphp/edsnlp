import json
import os
import subprocess
import textwrap
from pathlib import Path

from edsnlp.tune import tune

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(cmd, cwd: Path, env=None):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def get_local_tiny_bert_path() -> Path:
    snapshots = sorted(
        (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--hf-internal-testing--tiny-bert"
            / "snapshots"
        ).glob("*/config.json")
    )
    if not snapshots:
        raise AssertionError("No local hf-internal-testing/tiny-bert snapshot found.")
    return snapshots[-1].parent


def write_repo_files(repo: Path, model_path: Path):
    (repo / "data").mkdir()
    (repo / "checkpoint").mkdir()
    (repo / "data" / "doc1.txt").write_text("Covid positif.\n", encoding="utf-8")
    (repo / "data" / "doc1.ann").write_text(
        "T1\tsosy 0 5\tCovid\n",
        encoding="utf-8",
    )
    (repo / "data" / "doc2.txt").write_text("Pas de Covid.\n", encoding="utf-8")
    (repo / "data" / "doc2.ann").write_text(
        "T1\tsosy 7 12\tCovid\n",
        encoding="utf-8",
    )

    (repo / "config.yml").write_text(
        textwrap.dedent(
            """
            nlp:
              "@core": pipeline
              lang: eds
              components:
                ner:
                  '@factory': eds.ner_crf
                  embedding:
                    '@factory': eds.transformer
                    model: __MODEL_PATH__
                    window: 16
                    stride: 8
                  target_span_getter: gold_spans
                  span_setter: ents
                  mode: joint
                  window: 2

            scorer:
              speed: false
              ner:
                '@metrics': eds.ner_exact
                span_getter: ${nlp.components.ner.target_span_getter}

            optimizer:
              "@core": optimizer
              optim: AdamW
              module: ${nlp}
              groups:
                ".*":
                  lr:
                    "@schedules": linear
                    start_value: 1e-4
                    max_value: 2e-4
                    warmup_rate: 0.0
              total_steps: ${train.max_steps}

            train_data:
              data:
                "@readers": standoff
                path: ./data
                converter:
                  - '@factory': eds.standoff_dict2doc
                    span_setter: gold_spans
              shuffle: dataset
              batch_size: 1 docs
              pipe_names: ["ner"]

            val_data:
              "@readers": standoff
              path: ./data
              converter:
                - '@factory': eds.standoff_dict2doc
                  span_setter: gold_spans

            train:
              nlp: ${nlp}
              logger: ["json"]
              output_dir: artifacts
              train_data: ${train_data}
              val_data: ${val_data}
              max_steps: 1
              validation_interval: 1
              scorer: ${scorer}
              optimizer: ${optimizer}
              num_workers: 0
              cpu: true
            """
        )
        .replace("__MODEL_PATH__", model_path.as_posix())
        .strip()
        + "\n",
        encoding="utf-8",
    )

    (repo / "dvc.yaml").write_text(
        textwrap.dedent(
            """
            stages:
              train:
                cmd: python -m edsnlp.train --config config.yml
                deps:
                  - data
                  - config.yml
                params:
                  - config.yml:
                      - train.max_steps
                metrics:
                  - artifacts/metrics.json:
                      cache: false
                outs:
                  - artifacts/model-last
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    (repo / ".gitignore").write_text(
        textwrap.dedent(
            """
            .dvc-site-cache
            .dvc-global-config
            checkpoint/
            results/
            """
        )
    )


def init_repo(repo: Path, env):
    run(["git", "init"], cwd=repo, env=env)
    run(["git", "config", "user.email", "tests@example.com"], cwd=repo, env=env)
    run(["git", "config", "user.name", "EDS NLP Tests"], cwd=repo, env=env)
    run(["dvc", "init"], cwd=repo, env=env)
    run(["git", "add", "."], cwd=repo, env=env)
    run(["git", "commit", "-m", "init"], cwd=repo, env=env)


def test_tune_dvc_in_temp_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), env["PYTHONPATH"]]
        if env.get("PYTHONPATH")
        else [str(PROJECT_ROOT)]
    )
    env["DVC_SITE_CACHE_DIR"] = str(repo / ".dvc-site-cache")
    env["DVC_GLOBAL_CONFIG_DIR"] = str(repo / ".dvc-global-config")
    env["DVC_NO_ANALYTICS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    write_repo_files(repo, get_local_tiny_bert_path())
    init_repo(repo, env)

    monkeypatch.chdir(repo)
    monkeypatch.setenv("PYTHONPATH", env["PYTHONPATH"])
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", env["DVC_SITE_CACHE_DIR"])
    monkeypatch.setenv("DVC_GLOBAL_CONFIG_DIR", env["DVC_GLOBAL_CONFIG_DIR"])
    monkeypatch.setenv("DVC_NO_ANALYTICS", env["DVC_NO_ANALYTICS"])
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", env["TOKENIZERS_PARALLELISM"])

    tune(
        config_meta={"config_path": ["config.yml"]},
        hyperparameters={
            "train.max_steps": {
                "alias": "max_steps",
                "type": "int",
                "low": 1,
                "high": 2,
            }
        },
        output_dir="results",
        checkpoint_dir="checkpoint",
        n_trials=1,
        execution="dvc",
        metric="ner.micro.f",
        parallel_trials=False,
        seed=42,
    )

    results_file = repo / "results" / "results_summary.txt"
    tuned_config = repo / "results" / "config.yml"
    exp_show = json.loads(
        run(["dvc", "exp", "show", "--json", "--no-pager"], cwd=repo, env=env).stdout
    )
    recorded_experiments = [
        experiment
        for baseline in exp_show
        for experiment in baseline.get("experiments") or []
        if experiment.get("name", "").startswith("optuna-")
    ]

    assert results_file.exists()
    assert tuned_config.exists()
    assert "Best trial" in results_file.read_text(encoding="utf-8")
    assert recorded_experiments
    assert recorded_experiments[0]["executor"]["state"] == "success"
    print(recorded_experiments)
