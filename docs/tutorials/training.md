# Training a Named Entity Recognition model

In this tutorial, we'll see how we can train a deep learning model with EDS-NLP.
We also recommend looking at an existing project as a reference, such as [eds-pseudo](https://github.com/eds-pseudo) or [mlnorm](https://github.com/percevalw/mlnorm).

!!! warning "Hardware requirements"

    Training a modern deep learning model requires a lot of computational resources. We recommend using a machine with a GPU, ideally with at least 16GB of VRAM. If you don't have access to a GPU, you can use a cloud service like [Google Colab](https://colab.research.google.com/), [Kaggle](https://www.kaggle.com/), [Paperspace](https://www.paperspace.com/) or [Vast.ai](https://vast.ai/).

If you need a high level of control over the training procedure, we suggest you read the next ["Custom training script"](../make-a-training-script) tutorial.

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
    "edsnlp[ml]>=0.13.0",
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

??? note "A word about Confit"

    EDS-NLP makes heavy use of [Confit](https://aphp.github.io/confit/), a configuration library that allows you call functions from Python or the CLI, and validate and optionally cast their arguments.

    The EDS-NLP function used in this script is the `train` function of the `edsnlp.train` module. When passing a dict to a type-hinted argument (either from a `config.cfg` file, or by calling the function in Python), Confit will instantiate the correct class with the arguments provided in the dict. For instance, we pass a dict to the `val_data` parameter, which is actually type hinted as a `SampleGenerator`. Therefore, you can also instantiate a `SampleGenerator` object directly and pass it to the function.

    You can also tell Confit specifically which class you want to instantiate by using the `@register_name = "name_of_the_registered_class"` key and value in a dict or config section. We make a heavy use of this mechanism to build pipeline architectures.

=== "From the command line"

    Create a `config.cfg` file in the `configs` folder with the following content:

    ```{ .toml title="configs/config.cfg" }
    # 🤖 PIPELINE DEFINITION

    [nlp]
    # Word-level tokenization: use the "eds" tokenizer
    lang = "eds"
    # Our pipeline will contain a single NER pipe
    pipeline = ["ner"]
    batch_size = 1
    components = ${components}

    # The NER pipe will be a CRF model
    [components.ner]
    @factory = "eds.ner_crf"
    mode = "joint"
    target_span_getter = ${vars.gold_span_group}
    # Set spans as both to ents and in separate `ent.label` groups
    span_setter = [ "ents", "*" ]
    infer_span_setter = true

    # The CRF model will use a CNN to re-contextualize embeddings
    [components.ner.embedding]
    @factory = "eds.text_cnn"
    kernel_sizes = [3]

    # The base embeddings will be computed by a transformer
    # with a sliding window to reduce memory usage, increase
    # speed and allow for sequences longer than 512 wordpieces
    [components.ner.embedding.embedding]
    @factory = "eds.transformer"
    model = "camembert-base"
    window = 128
    stride = 96

    # 📈 SCORERS

    # that we will use to evaluate our model
    [scorer.ner]
    @metrics = "eds.ner_exact"
    span_getter = ${vars.gold_span_group}

    # Some variables grouped here, we could also
    # put their values directly in the config
    [vars]
    train = "./data/dataset/train"
    dev = "./data/dataset/test"
    gold_span_group = "gold_spans"

    # 🚀 TRAIN SCRIPT OPTIONS
    # -> python -m edsnlp.train --config configs/config.cfg

    [train]
    nlp = ${nlp}
    max_steps = 2000
    validation_interval = ${train.max_steps//10}
    warmup_rate = 0.1
    # Adapt to the VRAM of your GPU
    grad_accumulation_max_tokens = 48000
    batch_size = 2000 words
    transformer_lr = 5e-5
    task_lr = 1e-4
    scorer = ${scorer}
    output_path = "artifacts/model-last"

    [train.train_data]
    randomize = true
    # Documents will be split into sub-documents of 384 words
    # at most, covering multiple sentences. This makes the
    # assumption that entities do not span more than 384 words.
    max_length = 384
    multi_sentence = true
    [train.train_data.reader]
    # In what kind of files (ie. their extensions) is our
    # training data stored
    @readers = "standoff"
    path = ${vars.train}
    # What schema is used in the data files
    converter = "standoff"  # by default when readers==standoff
    span_setter = ${vars.gold_span_group}

    [train.val_data]
    [train.val_data.reader]
    @readers = "standoff"
    path = ${vars.dev}
    span_setter = ${vars.gold_span_group}

    # 📦 PACKAGE SCRIPT OPTIONS
    # -> python -m edsnlp.package --config configs/config.cfg

    [package]
    pipeline = ${train.output_path}
    name = "my_ner_model"
    ```

    To train the model, you can use the following command:

    ```{ .bash data-md-color-scheme="slate" }
    python -m edsnlp.train --config configs/config.cfg --seed 42
    ```

    *Any option can also be set either via the CLI or in `config.cfg` under `[train]`.*

=== "From a script or a notebook"

    Create a notebook, with the following content:

    ```{ .python .no-check }
    import edsnlp
    from edsnlp.train import train
    from edsnlp.metrics.ner import NerExactMetric
    import edsnlp.pipes as eds

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
    train_data_reader = edsnlp.data.read_standoff(
        path="./data/dataset/train", span_setter="gold_spans"
    )
    val_data_reader = edsnlp.data.read_standoff(
        path="./data/dataset/test", span_setter="gold_spans"
    )

    # 🚀 TRAIN
    train(
        nlp=nlp,
        max_steps=2000,
        validation_interval=200,
        warmup_rate=0.1,
        # Adapt to the VRAM of your GPU
        grad_accumulation_max_tokens=48000,
        batch_size=2000,
        transformer_lr=5e-5,
        task_lr=1e-4,
        scorer={"ner": ner_metric},
        output_path="artifacts/model-last",
        train_data={
            "randomize": True,
            # Documents will be split into sub-documents of 384 words
            # at most, covering multiple sentences. This makes the
            # assumption that entities do not span more than 384 words.
            "max_length": 384,
            "multi_sentence": True,
            "reader": train_data_reader,
        },
        val_data={
            "reader": val_data_reader,
        },
    )
    ```

    or use the config file:

    ```{ .python .no-check }
    from edsnlp.train import train
    import edsnlp
    import confit

    cfg = confit.Config.from_disk(
        "configs/config.cfg", resolve=True, registry=edsnlp.registry
    )
    nlp = train(**cfg["train"])
    ```

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

*Parametrize either via the CLI or in `config.cfg` under `[package]`.*

Tthe model saved at the train script output path (`artifacts/model-last`) will be named `my_ner_model` and will be saved in the `dist` folder. You can upload it to a package registry or install it directly with

```{ .bash data-md-color-scheme="slate" }
pip install dist/my_ner_model-0.1.0.tar.gz
```
