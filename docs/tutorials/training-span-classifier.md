# Training a span classifier

In this tutorial, we’ll train a hybrid **biopsy date extraction** model with EDS-NLP using the `edsnlp.train` API. Our goal will be to distinguish **biopsy dates** from other dates. We’ll use a small, annotated dataset of dates to train the model, and then apply the model to the date candidates extracted by the rule-based `eds.dates` component.

!!! warning "Hardware requirements"

    Training modern deep-learning models is compute-intensive. A GPU with **≥ 16 GB VRAM** is recommended. Training on CPU is possible but much slower. On macOS, PyTorch’s MPS backend may not support all operations and you'll likely hit `NotImplementedError` messages : in this case, fall back to CPU using the `cpu=True` option.

This tutorial uses EDS-NLP’s command-line interface, `python -m edsnlp.train`. If you need fine-grained control over the loop, consider [**writing your own training script**](./make-a-training-script.md).

## Creating a project

If you already have `edsnlp[ml]` installed, skip to the [next section](#creating-the-dataset)

Create a new project:

```bash { data-md-color-scheme="slate" }
mkdir my_span_classification_project
cd my_span_classification_project

touch README.md pyproject.toml
mkdir -p configs
```

Add a `pyproject.toml`:

```toml { title="pyproject.toml" }
[project]
name = "my_span_classification_project"
version = "0.1.0"
description = ""
authors = [
    { name = "Firstname Lastname", email = "firstname.lastname@domain.com" }
]
readme = "README.md"
requires-python = ">3.10,<4.0"

dependencies = [
    "edsnlp[ml]>=0.16.0",
    "sentencepiece>=0.1.96"
]

[dependency-groups]
dev = [
    "dvc>=2.37.0; python_version >= '3.8'",
    "pandas>=1.4.0,<2.0.0; python_version >= '3.8'",
    "pre-commit>=2.18.1",
    "accelerate>=0.21.0; python_version >= '3.8'",
    "rich-logger>=0.3.0"
]
```

We recommend using a virtual environment and [uv](https://docs.astral.sh/uv/):

```bash { data-md-color-scheme="slate" }
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install -e . --group dev
```

## Creating the dataset

We'll use a small dataset of annotated biopsy dates. The dataset is in the [standoff format](https://brat.nlplab.org/standoff). You can use [Brat](https://brat.nlplab.org/) to visualize and edit the annotations.
The dataset is available under [`tests/training/dataset_2`](https://github.com/aphp/edsnlp/tree/master/tests/training/dataset_2) directory of EDS-NLP's repository.

To use it, download and copy it into a local `dataset` directory:

- You can clone the repository and copy it yourself, or
- Use this direct downloader [link](https://download-directory.github.io/?url=https%3A%2F%2Fgithub.com%2Faphp%2Fedsnlp%2Ftree%2Fmaster%2Ftests%2Ftraining%2Fdataset) and unzip the downloaded archive.

## Training the model

You can train the model either from the command line or from a script or a notebook.
Visit the [`edsnlp.train` documentation][edsnlp.training.trainer.train] for a list of all the available options.

=== "From the command line"

    Create a config file:

    ```yaml { title="configs/config.yml" }
    vars:
      train: './dataset/train'
      dev: './dataset/dev'

    # 🤖 PIPELINE DEFINITION
    nlp:
      '@core': pipeline
      lang: eds

      components:
        normalizer:
          '@factory': eds.normalizer

        sentences:
          '@factory': eds.sentences

        # When we encounter a new document, first we extract the dates from the text
        dates:
          '@factory': eds.dates
          span_setter: 'ents'  # (1)!

        # Then for each date, we classify it as a biopsy date or not
        biopsy_classifier:
          '@factory': eds.span_classifier
          attributes: [ "is_biopsy_date" ]
          # The qualifier will run on spans in both `doc.ents` and
          # `doc.spans['gold_spans']` but `gold_spans` will be empty at
          # inference (unless spans are added explicitely by an upstream
          # component) and `ents` will be empty during training since others
          # components don't automatically feed into trainable components.
          span_getter: [ "ents", "gold_spans" ]
          # For context we'll use the current sent limited to +/- 10 words window
          context_getter: sents[0:0] & words[-10:10]
          # ...embedded by pooling the embeddings
          embedding:
            '@factory': eds.span_pooler
            # ...of a transformer model
            embedding:
              '@factory': eds.transformer
              model: 'almanach/camembert-bio-base'
              window: 128
              stride: 96

    # 📈 SCORER
    scorer:
      biopsy_date:
        '@metrics': eds.span_attribute
        span_getter: ${nlp.components.biopsy_classifier.span_getter}
        qualifiers: ${nlp.components.biopsy_classifier.attributes}

    # 🎛️ OPTIMIZER
    optimizer:
      "@core": optimizer !draft  # (2)!
      optim: torch.optim.AdamW
      groups:
        # Small learning rate for the pretrained transformer model
        - selector: 'biopsy_classifier[.]embedding[.]embedding'
          lr:
            '@schedules': linear
            warmup_rate: 0.1
            start_value: 0.
            max_value: 5e-5
        # Larger learning rate for the rest of the model
        - selector: '.*'
          lr:
            '@schedules': linear
            warmup_rate: 0.1
            start_value: 3e-4
            max_value: 3e-4

    # 📚 DATA
    train_data:
      - data:
          # Load the training data from standoff (BRAT) files
          '@readers': standoff
          path: ${vars.train}
          converter:
            # Convert a standoff file to a Doc
            - '@factory': eds.standoff_dict2doc
              span_setter: 'gold_spans'
              span_attributes: [ 'is_biopsy_date' ]
              bool_attributes: [ 'is_biopsy_date' ]
            # Add sentences explicitly for context_getter during training
            - '@factory': eds.sentences
              nlp: ${nlp}
            # Split each doc into replicas, each with exactly one
            # span from the `gold_spans` group to improve mixing
            - '@factory': eds.explode
              span_getter: 'gold_spans'
        shuffle: dataset
        batch_size: 8 spans
        pipe_names: [ "biopsy_classifier" ]

    val_data:
      '@readers': standoff
      path: ${vars.dev}
      converter:
        '@factory': eds.standoff_dict2doc
        span_setter: 'gold_spans'
        span_attributes: [ 'is_biopsy_date' ]
        bool_attributes: [ 'is_biopsy_date' ]

    # 🚀 TRAIN SCRIPT OPTIONS
    train:
      nlp: ${nlp}
      train_data: ${train_data}
      val_data: ${val_data}
      max_steps: 250
      validation_interval: 50
      max_grad_norm: 1.0
      scorer: ${scorer}
      num_workers: 1
      output_dir: 'artifacts'
    ```

    1. Put entities extracted by `eds.dates` in `doc.ents`, instead of `doc.spans['dates']`.
    2. What does "draft" mean here ? We'll let the train function pass the nlp object
    to the optimizer after it has been been `post_init`'ed : `post_init` is the operation that
    looks at some data, finds how many label the model must learn, and updates the model weights
    to have as many heads as there are labels observed in the train data. This function will be
    called by `train`, so the optimizer should be defined *after*, when the model parameter
    tensors are final. To do that, instead of instantiating the optimizer right now, we create
    a "Draft", which will be instantiated inside the `train` function, once all the required
    parameters are set.

    And train the model:

    ```bash { data-md-color-scheme="slate" }
    python -m edsnlp.train --config configs/config.yml --seed 42
    ```

=== "From a script or a notebook"

    ```python { .no-check }
    import edsnlp
    from edsnlp.training import train, ScheduledOptimizer, TrainingData
    from edsnlp.metrics.span_attribute import SpanAttributeMetric
    import edsnlp.pipes as eds
    import torch

    # 🤖 PIPELINE DEFINITION
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.normalizer())
    nlp.add_pipe(eds.sentences())
    # When we encounter a new document, first we extract the dates from the text
    nlp.add_pipe(eds.dates(span_setter="ents"))  # (1)!
    # Then for each data, we classify the dates as biopsy dates or not
    nlp.add_pipe(
        eds.span_classifier(
            attributes=["is_biopsy_date"],
            span_getter=[
                "ents",  # used at inference time
                "gold_spans",  # used at training time
            ],
            # ...using the current sentence intersected with a +/- 5 words window
            context_getter="sents[0:0] & words[-5:5]",
            # ...embedded by pooling the embeddings
            embedding=eds.span_pooler(
                # ...of a transformer model
                embedding=eds.transformer(
                    model="almanach/camembert-bio-base",
                    window=128,
                    stride=96,
                ),
            ),
        ),
        name="biopsy_classifier",
    )

    # 📈 SCORER
    metric = SpanAttributeMetric(
        span_getter=nlp.pipes.biopsy_classifier.span_getter,
        qualifiers=nlp.pipes.biopsy_classifier.attributes,
    )

    # 📚 DATA
    train_docs = (
        edsnlp.data
        # Load and convert standoff files to Doc objects
        .read_standoff(
            "./dataset/train",
            span_setter="gold_spans",
            span_attributes=["is_biopsy_date"],
            bool_attributes=["is_biopsy_date"],
        )
        # Split each doc into replicas, each with exactly one
        # span from the `gold_spans` group to improve mixing
        .map(eds.explode(span_getter="gold_spans"))
        # Add sentences explicitly for context_getter during training
        .map(nlp.pipes.sentences)
    )
    val_docs = edsnlp.data.read_standoff(
        "./dataset/dev",
        span_setter="gold_spans",
        span_attributes=["is_biopsy_date"],
        bool_attributes=["is_biopsy_date"],
    )

    # 🎛️ OPTIMIZER (here it will be the same as thedefault one)
    optimizer = ScheduledOptimizer.draft(  # (2)!
        optim=torch.optim.AdamW,
        groups=[
            {
                "selector": "biopsy_classifier[.]embedding",
                "lr": {
                    "@schedules": "linear",
                    "warmup_rate": 0.1,
                    "start_value": 0.,
                    "max_value": 5e-5,
                },
            },
            {
                "selector": ".*",
                "lr": {
                    "@schedules": "linear",
                    "warmup_rate": 0.1,
                    "start_value": 3e-4,
                    "max_value": 3e-4,
                },
            },
        ]
    )

    # 🚀 TRAIN
    train(
        nlp=nlp,
        train_data=TrainingData(
            data=train_docs,
            batch_size="8 spans",
            pipe_names=["biopsy_classifier"],
            shuffle="dataset",
        ),
        val_data=val_docs,
        scorer={"biopsy_date": metric},
        optimizer=optimizer,
        max_steps=250,
        validation_interval=50,
        grad_max_norm=1.0,
        num_workers=0,
        output_dir="artifacts",
        # cpu=True,  # (optional) use CPU instead of GPU/MPS
    )
    ```

    1. Put entities extracted by `eds.dates` in `doc.ents`, instead of `doc.spans['dates']`.
    2. What does "draft" mean here ? We'll let the train function pass the nlp object
    to the optimizer after it has been been `post_init`'ed : `post_init` is the operation that
    looks at some data, finds how many label the model must learn, and updates the model weights
    to have as many heads as there are labels observed in the train data. This function will be
    called by `train`, so the optimizer should be defined *after*, when the model parameter
    tensors are final. To do that, instead of instantiating the optimizer right now, we create
    a "Draft", which will be instantiated inside the `train` function, once all the required
    parameters are set.


!!! note "Training data vs inference pipeline"

    Trainable components are trained independently, except for shared weights. During
    training, a component reads the annotations it needs from the `Doc` objects
    present in the training stream. It does not automatically reuse the outputs of
    upstream components in `nlp`, whether these components are trainable or not.

    In other words, the pipeline defines how annotations will be produced at
    inference time, but training docs are not automatically modified by upstream
    pipeline components.

    In this tutorial, `eds.dates` is part of the pipeline because it defines how
    candidate spans will be produced at inference time. During training, however, the
    span classifier still needs explicit input spans in the training docs, here
    provided under `gold_spans`. This is important because only these spans were
    annotated with the `is_biopsy_date` attribute.

    The same rule applies to auxiliary annotations used only to compute features. In
    the example below, the classifier uses `context_getter="sents[0:0] & words[-5:5]"`,
    so sentence boundaries are needed to build the context. They are produced by
    `eds.sentences` at inference time, but they must still be added explicitly to
    the training docs.

## Use the model

You can now load the trained pipeline and extract the biopsy dates it predicts:

```python { .no-check }
import edsnlp

nlp = edsnlp.load("artifacts/model-last")

docs = edsnlp.data.from_iterable([
    "Le 15/07/2023, un prélèvement a été réalisé."
])
docs = docs.map_pipeline(nlp)
docs.to_pandas(
    converter="ents",
    # by default, will list doc.ents
    span_attributes=["is_biopsy_date"],
)
```
