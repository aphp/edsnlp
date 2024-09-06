import json
from pathlib import Path
from typing import Dict

import torch
from confit import Cli

import edsnlp
from edsnlp.core.registries import registry
from edsnlp.train import GenericScorer, SampleGenerator

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="evaluate", registry=registry)
def evaluate(
    *,
    data: SampleGenerator,
    model_path: Path = "artifacts/model-last",
    scorer: GenericScorer,
    task_metadata: Dict = {},
):
    """
    Evaluate a model on a dataset. This function can be called from the command line or
    from a script.

    By default, the model is loaded from `artifacts/model-last`, and the results are
    stored both in `artifacts/test_metrics.json` and in the model's
    `artifacts/model-last/meta.json` file.

    Parameters
    ----------
    data: SampleGenerator
        A function that generates samples for evaluation
    model_path: Path
        The path to the model to evaluate
    scorer: GenericScorer
        A function that computes metrics on the model. You can also pass a dict:

        ```{ .python .no-check }
        scorer = {
            "ner": NerExactMetric(...),
            ...
        }
        ```
    task_metadata: Dict
        Metadata about the evaluation task. This will be stored in the model's meta.json
        file and is primarily meant to be parsed by the Hugging Face Hub, e.g.,
        but also to remove previous results for the same dataset.

        ```{ .python .no-check }
        task_metadata={
            "task": {"type": "token-classification"},
            "dataset": {
                "name": "my_dataset",
                "type": "private",
            },
        }
        ```

    Returns
    -------

    """
    nlp = edsnlp.load(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    val_docs = list(data(nlp))
    metrics = scorer(nlp, val_docs)

    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    Path("artifacts/test_metrics.json").write_text(json.dumps(metrics, indent=2))
    meta = json.loads((model_path / "meta.json").read_text())
    results = meta.setdefault("results", [])

    # Remove previous results for the same dataset
    dataset_name = task_metadata.get("dataset", {}).get("name")
    index = next(
        (
            i
            for i, res in enumerate(results)
            if res.get("dataset", {}).get("name") == dataset_name
        ),
        None,
    )
    if index is not None:
        results.pop(index)
    results.append(
        {
            "metrics": metrics,
            **task_metadata,
        }
    )
    (model_path / "meta.json").write_text(json.dumps(meta, indent=2))
    Path("artifacts/test_metrics.jsonl").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()
