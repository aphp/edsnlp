# Hyperparameter Tuning

In this tutorial, we'll see how we can quickly tune hyperparameters of a deep learning model with EDS-NLP using the `edsnlp.tune` function.

Tuning refers to the process of optimizing the hyperparameters of a machine learning model to achieve the best performance. These hyperparameters include factors like learning rate, batch size, dropout rates, and model architecture parameters. Tuning is crucial because the right combination of hyperparameters can significantly improve model accuracy and efficiency, while poor choices can lead to overfitting, underfitting, or unnecessary computational costs. By systematically searching for the best hyperparameters, we ensure the model is both effective and efficient before the final training phase.

We strongly suggest you read the previous ["Training API tutorial"](./training.md) to understand how to train a deep learning model using a config file with EDS-NLP.

## 1. Creating a project

If you already have installed `edsnlp[ml]` and do not want to setup a project, you can skip to the [next section](#tuning-the-model).

Create a new project:

```{ .bash data-md-color-scheme="slate" }
mkdir my_ner_project
cd my_ner_project

touch README.md pyproject.toml
mkdir -p configs data/dataset
```

Add a standard `pyproject.toml` file with the following content. This
file will be used to manage the dependencies of the project and its versioning.

```{ .toml title="pyproject.toml"}
[project]
name = "my_ner_project"
version = "0.1.0"
description = ""
authors = [
    { name="Firstname Lastname", email="firstname.lastname@domain.com" }
]
readme = "README.md"
requires-python = ">3.7.1,<4.0"

dependencies = [
    "edsnlp[ml]>=0.16.0",
    "sentencepiece>=0.1.96",
    "optuna>=4.0.0",
    "plotly>=5.18.0",
    "ruamel.yaml>=0.18.0",
    "configobj>=5.0.9",
]

[project.optional-dependencies]
dev = [
    "dvc>=2.37.0; python_version >= '3.8'",
    "pandas>=1.1.0,<2.0.0; python_version < '3.8'",
    "pandas>=1.4.0,<2.0.0; python_version >= '3.8'",
    "pre-commit>=2.18.1",
    "accelerate>=0.21.0; python_version >= '3.8'",
    "rich-logger>=0.3.0"
]
```

We recommend using a virtual environment ("venv") to isolate the dependencies of your project and using [uv](https://docs.astral.sh/uv/) to install the dependencies:

```{ .bash data-md-color-scheme="slate" }
pip install uv
# skip the next two lines if you do not want a venv
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]" -p $(uv python find)
```

## 2. Tuning a model

### 2.1. Tuning Section in `config.yml` file

If you followed the ["Training API tutorial"](./training.md), you should already have a `configs/config.yml` file for training parameters.

To enable hyperparameter tuning, add the following `tuning` section to your `config.yml` file:

```{ .yaml title="configs/config.yml" }
tuning:
  # Output directory for tuning results.
  output_dir: 'results'
  # Checkpoint directory
  checkpoint_dir: 'checkpoint'
  # Number of gpu hours allowed for tuning.
  gpu_hours: 1.0
  # Number of fixed trials to tune hyperparameters (override gpu_hours).
  n_trials: 4
  # Enable two-phase tuning. In the first phase, the script will tune all hyperparameters.
  # In the second phase, it will focus only on the top 50% most important hyperparameters.
  two_phase_tuning: True
  # Metric used to evaluate trials.
  metric: "ner.micro.f"
  # Hyperparameters to tune.
  hyperparameters:
```

Let's detail the new parameters:

- `output_dir`: Directory where tuning results, visualizations, and best parameters will be saved.
- `checkpoint_dir`: Directory where the tuning checkpoint will be saved each trial. Allows resuming previous tuning in case of a crash.
- `gpu_hours`: Estimated total GPU time available for tuning, in hours. Given this time, the script will automatically compute for how many training trials we can tune hyperparameters. By default, `gpu_hours` is set to 1.
- `n_trials`: Number of training trials for tuning. If provided, it will override `gpu_hours` and tune the model for exactly `n_trial` trials.
- `two_phase_tuning`: If True, performs a two-phase tuning. In the first phase, all hyperparameters are tuned, and in the second phase, the top half (based on importance) are fine-tuned while freezing others. By default, `two_phase_tuning` is False.
- `metric`: Metric used to evaluate trials. It corresponds to a path in the scorer results (depending on the scorer used in the config). By default `metric` is set to "ner.micro.f".
- `hyperparameters`: The list of hyperparameters to tune and details about their tunings. We will discuss how it work in the following section.

### 2.2. Add hyperparameters to tune

In the `config.yml` file, the `tuning.hyperparameters` section defines the hyperparameters to optimize. Each hyperparameter can be specified with its type, range, and additional properties. To add a hyperparameter, follow this syntax:

```{ .yaml title="configs/config.yml" }
tuning:
  hyperparameters:
    # Hyperparameter path in `config.yml`.
    "nlp.components.ner.embedding.embedding.classifier_dropout":
      # Alias name. If not specified, full path will be the name.
      alias: "classifier_dropout"
      # Type of the hyperparameter: 'int', 'float', or 'categorical'.
      type: "float"
      # Lower bound for tuning.
      low: 0.
      # Upper bound for tuning.
      high: 0.3
      # Step for discretization (optional).
      step: 0.05
```

Since `edsnlp.tune` leverages the [Optuna](https://optuna.org/) framework, we recommend reviewing the following Optuna functions to understand the properties you can specify for hyperparameter sampling:

- [suggest_float](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float) – For sampling floating-point hyperparameters.
- [suggest_int](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int) – For sampling integer hyperparameters.
- [suggest_categorical](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical) – For sampling categorical hyperparameters.

These resources provide detailed guidance on defining the sampling ranges, distributions, and additional properties for each type of hyperparameter.


### 2.3. Complete Example

Now, let's look at a complete example. Assume that we want to perform a two-phase tuning, for 40 gpu hours, on the following hyperparameters:

- `hidden_dropout_prob`: Dropout probability for hidden layers.
- `attention_dropout_prob`: Dropout probability for attention layers.
- `classifier_dropout`: Dropout probability for the classifier layer.
- `transformer_start_value`: Learning rate start value for the transformer.
- `transformer_max_value`: Maximum learning rate for the transformer.
- `transformer_warmup_rate`: Warmup rate for the transformer learning rate scheduler.
- `transformer_weight_decay`: Weight decay for the transformer optimizer.
- `other_start_value`: Learning rate start value for other components.
- `other_max_value`: Maximum learning rate for other components.
- `other_warmup_rate`: Warmup rate for the learning rate scheduler of other components.
- `other_weight_decay`: Weight decay for the optimizer of other components.

Then the full `config.yml` will be:

```{ .yaml title="configs/config.yml" }
vars:
  train: './data/dataset/train'
  dev: './data/dataset/test'

# 🤖 PIPELINE DEFINITION
nlp:
  '@core': pipeline
  lang: eds  # Word-level tokenization: use the "eds" tokenizer
  components:
    ner:
      '@factory': eds.ner_crf
      mode: 'joint'
      target_span_getter: 'gold_spans'
      span_setter: [ "ents", "*" ]
      infer_span_setter: true
      embedding:
        '@factory': eds.text_cnn
        kernel_sizes: [ 3 ]
        embedding:
          '@factory': eds.transformer
          model: prajjwal1/bert-tiny
          ignore_mismatched_sizes: True
          window: 128
          stride: 96
          # Dropout parameters passed to the underlying transformer object.
          hidden_dropout_prob: 0.1
          attention_probs_dropout_prob: 0.1
          classifier_dropout: 0.1

# 📈 SCORERS
scorer:
  ner:
    '@metrics': eds.ner_token
    span_getter: ${ nlp.components.ner.target_span_getter }

# 🎛️ OPTIMIZER
optimizer:
  "@core": optimizer
  optim: adamw
  groups:
    "^transformer":
      weight_decay: 1e-3
      lr:
        '@schedules': linear
        "warmup_rate": 0.1
        "start_value": 1e-5
        "max_value": 8e-5
    ".*":
      weight_decay: 1e-3
      lr:
        '@schedules': linear
        "warmup_rate": 0.1
        "start_value": 1e-5
        "max_value": 8e-5
  module: ${ nlp }
  total_steps: ${ train.max_steps }

# 📚 DATA
train_data:
  - data:
      '@readers': standoff
      path: ${ vars.train }
      converter:
        - '@factory': eds.standoff_dict2doc
          span_setter: 'gold_spans'
        - '@factory': eds.split
          nlp: null
          max_length: 256
          regex: '\n\n+'
    shuffle: dataset
    batch_size: 32 * 128 tokens
    pipe_names: [ "ner" ]

val_data:
  '@readers': standoff
  path: ${ vars.dev }
  converter:
    - '@factory': eds.standoff_dict2doc
      span_setter: 'gold_spans'

# 🚀 TRAIN SCRIPT OPTIONS
# -> python -m edsnlp.train --config configs/config.yml
train:
  nlp: ${ nlp }
  logger: True
  output_dir: 'artifacts'
  train_data: ${ train_data }
  val_data: ${ val_data }
  max_steps: 400
  validation_interval: ${ train.max_steps//2 }
  max_grad_norm: 1.0
  scorer: ${ scorer }
  optimizer: ${ optimizer }
  num_workers: 2

# 📦 PACKAGE SCRIPT OPTIONS
# -> python -m edsnlp.package --config configs/config.yml
package:
  pipeline: ${ train.output_dir }
  name: 'my_ner_model'

# ⚙️ TUNE SCRIPT OPTIONS
# -> python -m edsnlp.tune --config configs/config.yml
tuning:
  output_dir: 'results'
  checkpoint_dir: 'checkpoint'
  gpu_hours: 40.0
  two_phase_tuning: True
  metric: "ner.micro.f"
  hyperparameters:
    "nlp.components.ner.embedding.embedding.hidden_dropout_prob":
      alias: "hidden_dropout"
      type: "float"
      low: 0.
      high: 0.3
      step: 0.05
    "nlp.components.ner.embedding.embedding.attention_probs_dropout_prob":
      alias: "attention_dropout"
      type: "float"
      low: 0.
      high: 0.3
      step: 0.05
    "nlp.components.ner.embedding.embedding.classifier_dropout":
      alias: "classifier_dropout"
      type: "float"
      low: 0.
      high: 0.3
      step: 0.05
    "optimizer.groups.^transformer.lr.start_value":
      alias: "transformer_start_value"
      type: "float"
      low: 1e-6
      high: 1e-3
      log: True
    "optimizer.groups.^transformer.lr.max_value":
      alias: "transformer_max_value"
      type: "float"
      low: 1e-6
      high: 1e-3
      log: True
    "optimizer.groups.^transformer.lr.warmup_rate":
      alias: "transformer_warmup_rate"
      type: "float"
      low: 0.
      high: 0.3
      step: 0.05
    "optimizer.groups.^transformer.weight_decay":
      alias: "transformer_weight_decay"
      type: "float"
      low: 1e-4
      high: 1e-2
      log: True
    "optimizer.groups.'.*'.lr.warmup_rate":
      alias: "other_warmup_rate"
      type: "float"
      low: 0.
      high: 0.3
      step: 0.05
    "optimizer.groups.'.*'.lr.start_value":
      alias: "other_start_value"
      type: "float"
      low: 1e-6
      high: 1e-3
      log: True
    "optimizer.groups.'.*'.lr.max_value":
      alias: "other_max_value"
      type: "float"
      low: 1e-6
      high: 1e-3
      log: True
    "optimizer.groups.'.*'.weight_decay":
      alias: "other_weight_decay"
      type: "float"
      low: 1e-4
      high: 1e-2
      log: True
```

Finally, to lauch the tuning process, use the following command:

```{ .bash data-md-color-scheme="slate" }
python -m edsnlp.tune --config configs/config.yml --seed 42
```

## 3. Results

At the end of the tuning process, `edsnlp.tune` generates various results and saves them in the `output_dir` specified in the `config.yml` file:

- **Tuning Summary**: `result_summary.txt`, a summary file containing details about the best training trial, the best overall metric, the optimal hyperparameter values, and the average importance of each hyperparameter across all trials.
- **Optimal Configuration**: `config.yml`, containing the best hyperparameter values. Warning: Since the Confit library does not preserve style and comments, these will be lost in the resulting configuration file. If you need to retain them, manually update your original `config.yml` using the information from `result_summary.txt`.
- **Graphs and Visualizations**: Various graphics illustrating the tuning process, such as:
  - [**Optimization History plot**](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_optimization_history.html#sphx-glr-reference-visualization-generated-optuna-visualization-plot-optimization-history-py): A line graph showing the performance of each trial over time, illustrating the optimization process and how the model's performance improves with each iteration.
  - [**Empirical Distribution Function (EDF) plot**](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_edf.html#sphx-glr-reference-visualization-generated-optuna-visualization-plot-edf-py): A graph showing the cumulative distribution of the results, helping you understand the distribution of performance scores and providing insights into the variability and robustness of the tuning process.
  - [**Contour plot**](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_contour.html#sphx-glr-reference-visualization-generated-optuna-visualization-plot-contour-py): A 2D plot that shows the relationship between two hyperparameters and their combined effect on the objective metric, providing a clear view of the optimal parameter regions.
  - [**Parallel Coordinate plot**](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_parallel_coordinate.html#sphx-glr-reference-visualization-generated-optuna-visualization-plot-parallel-coordinate-py): A multi-dimensional plot where each hyperparameter is represented as a vertical axis, and each trial is displayed as a line connecting the hyperparameter values, helping you analyze correlations and patterns across hyperparameters and their impact on performance.
  - [**Timeline plot**](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_timeline.html#sphx-glr-reference-visualization-generated-optuna-visualization-plot-timeline-py): A 2D plot that displays all trials and their statuses ("completed," "pruned," or "failed") over time, providing a clear overview of the progress and outcomes of the tuning process.

These outputs offer a comprehensive view of the tuning results, enabling you to better understand the optimization process and easily deploy the best configuration.

**Note**: If you enabled two-phase tuning, the `output_dir` will contain two subdirectories, `phase_1` and `phase_2`, each with their own result files as described earlier. This separation allows you to analyze the results from each phase individually.

## 4. Final Training

Now that the hyperparameters have been tuned, you can update your final `config.yml` with the best-performing hyperparameters and proceed to launch the final training using the ["Training API"](./training.md).
