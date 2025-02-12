# Training API

In this tutorial, we'll see how we can quickly train a deep learning model with EDS-NLP using the `edsnlp.train` function.

!!! warning "Hardware requirements"

    Training a modern deep learning model requires a lot of computational resources. We recommend using a machine with a GPU, ideally with at least 16GB of VRAM. If you don't have access to a GPU, you can use a cloud service like [Google Colab](https://colab.research.google.com/), [Kaggle](https://www.kaggle.com/), [Paperspace](https://www.paperspace.com/) or [Vast.ai](https://vast.ai/).

If you need a high level of control over the training procedure, we suggest you read the previous ["Deep learning tutorial"](./make-a-training-script.md) to understand how to build a training loop from scratch with EDS-NLP.

## Creating a project

If you already have installed `edsnlp[ml]` and do not want to setup a project, you can skip to the [next section](#training-the-model).

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
    "edsnlp[ml]>=0.15.0",
    "sentencepiece>=0.1.96"
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

## Training the model

EDS-NLP supports training models either [from the command line](#from-the-command-line) or [from a Python script or notebook](#from-a-script-or-a-notebook), and switching between the two is straightforward thanks to the use of [Confit](https://aphp.github.io/confit/).

??? note "A word about Confit"

    EDS-NLP makes heavy use of [Confit](https://aphp.github.io/confit/), a configuration library that allows you call functions from Python or the CLI, and validate and optionally cast their arguments.

    The EDS-NLP function used in this script is the `train` function of the `edsnlp.train` module. When passing a dict to a type-hinted argument (either from a `config.yml` file, or by calling the function in Python), Confit will instantiate the correct class with the arguments provided in the dict. For instance, we pass a dict to the `val_data` parameter, which is actually type hinted as a `SampleGenerator`: this dict will actually be used as keyword arguments to instantiate this `SampleGenerator` object. You can also instantiate a `SampleGenerator` object directly and pass it to the function.

    You can also tell Confit specifically which class you want to instantiate by using the `@register_name = "name_of_the_registered_class"` key and value in a dict or config section. We make a heavy use of this mechanism to build pipeline architectures.

=== "From the command line"

    Create a `config.yml` file in the `configs` folder with the following content:

    ```{ .yaml title="configs/config.yml" }
    # Some variables are grouped here for conviency but we could also
    # put their values directly in the config in place of their reference
    vars:
      train: './data/dataset/train'
      dev: './data/dataset/test'

    # 🤖 PIPELINE DEFINITION
    nlp:
      '@core': pipeline  #(1)!
      lang: eds  # Word-level tokenization: use the "eds" tokenizer

      # Our pipeline will contain a single NER pipe
      # The NER pipe will be a CRF model
      components:
        ner:
          '@factory': eds.ner_crf
          mode: 'joint'
          target_span_getter: 'gold_spans'
          # Set spans as both to ents and in separate `ent.label` groups
          span_setter: [ "ents", "*" ]
          infer_span_setter: true

          # The CRF model will use a CNN to re-contextualize embeddings
          embedding:
            '@factory': eds.text_cnn
            kernel_sizes: [ 3 ]

            # The base embeddings will be computed by a transformer
            embedding:
              '@factory': eds.transformer
              model: 'camembert-base'
              window: 128
              stride: 96

    # 📈 SCORERS
    scorer:
      ner:
        '@metrics': eds.ner_exact
        span_getter: ${ nlp.components.ner.target_span_getter }

    # 🎛️ OPTIMIZER
    optimizer:
      "@core": optimizer
      optim: adamw
      groups:
        # Assign parameters starting with transformer (ie the parameters of the transformer component)
        # to a first group
        "^transformer":
          lr:
            '@schedules': linear
            "warmup_rate": 0.1
            "start_value": 0
            "max_value": 5e-5
        # And every other parameters to the second group
        "":
          lr:
            '@schedules': linear
            "warmup_rate": 0.1
            "start_value": 3e-4
            "max_value": 3e-4
      module: ${ nlp }
      total_steps: ${ train.max_steps }

    # 📚 DATA
    train_data:
      - data:
          # In what kind of files (ie. their extensions) is our
          # training data stored
          '@readers': standoff
          path: ${ vars.train }
          converter:
            # What schema is used in the data files
            - '@factory': eds.standoff_dict2doc
              span_setter: 'gold_spans'
            # How to preprocess each doc for training
            - '@factory': eds.split
              nlp: null
              max_length: 2000
              regex: '\n\n+'
        shuffle: dataset
        batch_size: 4096 tokens  # 32 * 128 tokens
        pipe_names: [ "ner" ]

    val_data:
      '@readers': standoff
      path: ${ vars.dev }
      # What schema is used in the data files
      converter:
        - '@factory': eds.standoff_dict2doc
          span_setter: 'gold_spans'

    # 🚀 TRAIN SCRIPT OPTIONS
    # -> python -m edsnlp.train --config configs/config.yml
    train:
      nlp: ${ nlp }
      output_dir: 'artifacts'
      train_data: ${ train_data }
      val_data: ${ val_data }
      max_steps: 2000
      validation_interval: ${ train.max_steps//10 }
      grad_max_norm: 1.0
      scorer: ${ scorer }
      optimizer: ${ optimizer }
      # Do preprocessing in parallel on 1 worker
      num_workers: 1
      # Enable on Mac OS X or if you don't want to use available GPUs
      # cpu: true

    # 📦 PACKAGE SCRIPT OPTIONS
    # -> python -m edsnlp.package --config configs/config.yml
    package:
      pipeline: ${ train.output_dir }
      name: 'my_ner_model'
    ```

    1. Why do we use `'@core': pipeline` here ? Because we need the reference used in `optimizer.module = ${ nlp }` to be the actual Pipeline and not its keyword arguments : when confit sees `'@core': pipeline`, it will instantiate the `Pipeline` class with the arguments provided in the dict.

        In fact, you could also use `'@core': eds.pipeline` in every config when you define a pipeline, but sometimes it's more convenient to let Confit infer that the type of the nlp argument based on the function when it's type hinted. Not specifying `'@core': pipeline` is also more aligned with `spacy`'s pipeline config API. However, in general, explicit is better than implicit, so feel free to use explicitly write `'@core': eds.pipeline` when you define a pipeline.

    To train the model, you can use the following command:

    ```{ .bash data-md-color-scheme="slate" }
    python -m edsnlp.train --config configs/config.yml --seed 42
    ```

    *Any option can also be set either via the CLI or in `config.yml` under `[train]`.*

=== "From a script or a notebook"

    Create a notebook, with the following content:

    ```{ .python .no-check }
    import edsnlp
    from edsnlp.training import train, ScheduledOptimizer, TrainingData
    from edsnlp.metrics.ner import NerExactMetric
    import edsnlp.pipes as eds
    import torch

    # 🤖 PIPELINE DEFINITION
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        # The NER pipe will be a CRF model
        eds.ner_crf(
            mode="joint",
            target_span_getter="gold_spans",
            # Set spans as both to ents and in separate `ent.label` groups
            span_setter=["ents", "*"],
            infer_span_setter=True,
            # The CRF model will use a CNN to re-contextualize embeddings
            embedding=eds.text_cnn(
                kernel_sizes=[3],
                # The base embeddings will be computed by a transformer
                embedding=eds.transformer(
                    model="camembert-base",
                    window=128,
                    stride=96,
                ),
            ),
        )
    )

    # 📈 SCORERS
    ner_metric = NerExactMetric(span_getter="gold_spans")

    # 📚 DATA
    train_data = (
        edsnlp.data
        .read_standoff("./data/dataset/train", span_setter="gold_spans")
        .map(eds.split(nlp=None, max_length=2000, regex="\n\n+"))
    )
    val_data = (
        edsnlp.data
        .read_standoff("./data/dataset/test", span_setter="gold_spans")
    )

    # 🎛️ OPTIMIZER
    max_steps = 2000
    optimizer = ScheduledOptimizer(
        optim=torch.optim.Adam,
        module=nlp,
        total_steps=max_steps,
        groups={
            "^transformer": {
                "lr": {"@schedules": "linear", "warmup_rate": 0.1, "start_value": 0 "max_value": 5e-5,},
            },
            "": {
                "lr": {"@schedules": "linear", "warmup_rate": 0.1, "start_value": 3e-4 "max_value": 3e-4,},
            },
        },
    )

    # 🚀 TRAIN
    train(
        nlp=nlp,
        max_steps=max_steps,
        validation_interval=max_steps // 10,
        train_data=TrainingData(
            data=train_data,
            batch_size="4096 tokens",  # 32 * 128 tokens
            pipe_names=["ner"],
            shuffle="dataset",
        ),
        val_data=val_data,
        scorer={"ner": ner_metric},
        optimizer=optimizer,
        grad_max_norm=1.0,
        output_dir="artifacts",
        # Do preprocessing in parallel on 1 worker
        num_workers=1,
        # Enable on Mac OS X or if you don't want to use available GPUs
        # cpu=True,
    )
    ```

or use the config file:

```{ .python .no-check }
from edsnlp.train import train
import edsnlp
import confit

cfg = confit.Config.from_disk(
    "configs/config.yml", resolve=True, registry=edsnlp.registry
)
nlp = train(**cfg["train"])
```

Here are the parameters you can pass to the `train` function:

::: edsnlp.training.trainer.train
    options:
        heading_level: 4
        only_parameters: true
        skip_parameters: []
        show_source: false
        show_toc: false

## Use the model

You can now load the model and use it to process some text:

```{ .python .no-check }
import edsnlp

nlp = edsnlp.load("artifacts/model-last")
doc = nlp("Some sample text")
for ent in doc.ents:
    print(ent, ent.label_)
```

## Packaging the model

To package the model and share it with friends or family (if the model does not contain sensitive data), you can use the following command:

```{ .bash data-md-color-scheme="slate" }
python -m edsnlp.package --pipeline artifacts/model-last/ --name my_ner_model --distributions sdist
```

*Parametrize either via the CLI or in `config.yml` under `[package]`.*

Tthe model saved at the train script output path (`artifacts/model-last`) will be named `my_ner_model` and will be saved in the `dist` folder. You can upload it to a package registry or install it directly with

```{ .bash data-md-color-scheme="slate" }
pip install dist/my_ner_model-0.1.0.tar.gz
```
