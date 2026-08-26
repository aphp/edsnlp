# Training a document classifier

In this tutorial, we'll train **document-level classifiers** with EDS-NLP using the `edsnlp.train` API. Unlike a NER or a span classifier, which label *parts* of a document, a document classifier reads the whole note and predicts an attribute of the document itself: its type, the diagnoses it should be coded with, the topics it covers.

We'll go through two use cases:

- a **single head**, predicting the type of a document — the most common case ;
- **several heads** sharing a single document embedding, to code a hospital stay with its principal diagnosis (DP) and its associated diagnoses (DAS).

!!! warning "Hardware requirements"

    Training modern deep-learning models is compute-intensive. A GPU with **≥ 16 GB VRAM** is recommended. Training on CPU is possible but much slower. On macOS, PyTorch's MPS backend may not support all operations and you'll likely hit `NotImplementedError` messages : in this case, fall back to CPU using the `cpu=True` option.

This tutorial uses EDS-NLP's command-line interface, `python -m edsnlp.train`. If you need fine-grained control over the loop, consider [**writing your own training script**](./make-a-training-script.md).

## Creating a project

If you already have `edsnlp[ml]` installed, skip to the [next section](#creating-the-dataset)

Create a new project:

```bash { data-md-color-scheme="slate" }
mkdir my_doc_classification_project
cd my_doc_classification_project

touch README.md pyproject.toml
mkdir -p configs
```

Add a `pyproject.toml`:

```toml { title="pyproject.toml" }
[project]
name = "my_doc_classification_project"
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

Document-level labels don't fit the [standoff format](https://brat.nlplab.org/standoff), which annotates offsets inside a text. We'll store them next to the text instead, in a JSONL file — one JSON object per line — which `edsnlp.data.read_json` reads and the `eds.omop_dict2doc` converter turns into `Doc` objects:

```json { title="dataset/doc_types_train.jsonl" }
{"note_id": "1", "note_text": "CONSULTATION DU 07/03/2025\nMotif : réévaluation d'une dyspnée d'effort. …", "doc_type": "compte_rendu_consultation"}
{"note_id": "2", "note_text": "RÉSULTATS DE BIOLOGIE — prélèvement du 14/05/2025\nHémogramme : hémoglobine 14,9 g/dL …", "doc_type": "compte_rendu_biologie"}
```

Any column listed in `doc_attributes` is copied to the matching `Doc._` extension, so `doc_type` above becomes `doc._.doc_type`. The same mechanism works for a *list* of labels, which is what we'll use in the second part of this tutorial.

We'll use a small synthetic corpus of French clinical notes, available under the [`tests/training/dataset_3`](https://github.com/aphp/edsnlp/tree/master/tests/training/dataset_3) directory of EDS-NLP's repository. To use it, download and copy it into a local `dataset` directory:

- You can clone the repository and copy it yourself, or
- Use this direct downloader [link](https://download-directory.github.io/?url=https%3A%2F%2Fgithub.com%2Faphp%2Fedsnlp%2Ftree%2Fmaster%2Ftests%2Ftraining%2Fdataset_3) and unzip the downloaded archive.

It holds five files, one train/dev pair per use case:

| File | Documents | Annotated with |
|---|---|---|
| `doc_types_train.jsonl` | 60 | `doc_type` |
| `doc_types_dev.jsonl` | 16 | `doc_type` |
| `coding_train.jsonl` | 60 | `dp`, `das` |
| `coding_dev.jsonl` | 15 | `dp`, `das` |
| `coding_dp_only.jsonl` | 30 | `dp` only — used to illustrate partial supervision |

!!! warning "A toy corpus"

    These notes are synthetic and were generated from a handful of templates, so a model will
    fit them very quickly and reach scores that mean nothing. They are here to make the code
    below runnable end to end, not to benchmark anything.

!!! note "Other formats"

    JSONL is convenient, but nothing here is specific to it: any [reader](../data/index.md) producing dicts with a `note_text` key will do, including `edsnlp.data.read_parquet` and `edsnlp.data.from_pandas`. On a large corpus, Parquet is usually the better choice.

## A single head: predicting the document type

A document has exactly one type, so this is a **single-label** problem: we use one `eds.single_label_head`, keyed by the name of the attribute it fills in.

Note that we do not list the labels: left out, they are inferred from the training data when `nlp.post_init(...)` is called by `train`. Pass `labels=[...]` explicitly if you'd rather pin them down — or a path to a pickled list, which is handier when there are thousands of them.

=== "From the command line"

    Create a config file:

    ```yaml { title="configs/doc_type.yml" }
    vars:
      train: './dataset/doc_types_train.jsonl'
      dev: './dataset/doc_types_dev.jsonl'

    # 🤖 PIPELINE DEFINITION
    nlp:
      '@core': pipeline
      lang: eds

      components:
        doc_classifier:
          '@factory': eds.doc_classifier
          # The whole document is squeezed into a single vector...
          embedding:
            '@factory': eds.doc_pooler
            pooling_mode: 'mean'  # (1)!
            # ...by averaging the word embeddings of a transformer model
            embedding:
              '@factory': eds.transformer
              model: 'almanach/camembert-bio-base'
              window: 128
              stride: 96
          # ...and fed to a single head, which writes to `doc._.doc_type`
          heads:
            doc_type:
              '@misc': eds.single_label_head
              loss: 'ce'
              dropout_rate: 0.1

    # 📈 SCORER
    scorer:
      classif:
        '@metrics': eds.doc_classification
        label_attr: [ 'doc_type' ]

    # 🎛️ OPTIMIZER
    optimizer:
      "@core": optimizer !draft  # (2)!
      optim: torch.optim.AdamW
      groups:
        # Small learning rate for the pretrained transformer model
        - selector: 'doc_classifier[.]embedding'
          lr:
            '@schedules': linear
            warmup_rate: 0.1
            start_value: 0.
            max_value: 5e-5
        # Larger learning rate for the head
        - selector: '.*'
          lr:
            '@schedules': linear
            warmup_rate: 0.1
            start_value: 3e-4
            max_value: 3e-4

    # 📚 DATA
    train_data:
      - data:
          '@readers': json
          path: ${vars.train}
          converter:
            - '@factory': eds.omop_dict2doc
              # Copy the `doc_type` column to `doc._.doc_type`
              doc_attributes: [ 'doc_type' ]
        shuffle: dataset
        batch_size: 8 docs
        pipe_names: [ "doc_classifier" ]

    val_data:
      '@readers': json
      path: ${vars.dev}
      converter:
        - '@factory': eds.omop_dict2doc
          doc_attributes: [ 'doc_type' ]

    # 🚀 TRAIN SCRIPT OPTIONS
    train:
      nlp: ${nlp}
      train_data: ${train_data}
      val_data: ${val_data}
      max_steps: 400
      validation_interval: 100
      max_grad_norm: 1.0
      scorer: ${scorer}
      num_workers: 1
      output_dir: 'artifacts'
    ```

    1. `eds.doc_pooler` supports `mean`, `max`, `sum`, `attention` and `cls`. `mean` is a solid
    default ; `attention` learns which words matter and usually helps on long notes ; `cls`
    reuses the transformer's own `[CLS]` vector, and therefore requires `eds.transformer` as the
    underlying embedding.
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
    python -m edsnlp.train --config configs/doc_type.yml --seed 42
    ```

=== "From a script or a notebook"

    ```python { .no-check }
    import edsnlp
    import edsnlp.pipes as eds
    import torch
    from edsnlp.metrics.doc_classification import DocClassificationMetric
    from edsnlp.pipes.trainable.doc_classifier.heads import SingleLabelHead
    from edsnlp.training import ScheduledOptimizer, TrainingData, train

    # 🤖 PIPELINE DEFINITION
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.doc_classifier(
            # The whole document is squeezed into a single vector...
            embedding=eds.doc_pooler(
                pooling_mode="mean",  # (1)!
                # ...by averaging the word embeddings of a transformer model
                embedding=eds.transformer(
                    model="almanach/camembert-bio-base",
                    window=128,
                    stride=96,
                ),
            ),
            # ...and fed to a single head, which writes to `doc._.doc_type`
            heads={
                "doc_type": SingleLabelHead(loss="ce", dropout_rate=0.1),
            },
        ),
        name="doc_classifier",
    )

    # 📈 SCORER
    metric = DocClassificationMetric(label_attr=["doc_type"])

    # 📚 DATA
    train_docs = edsnlp.data.read_json(
        "./dataset/doc_types_train.jsonl",
        converter="omop",
        # Copy the `doc_type` column to `doc._.doc_type`
        doc_attributes=["doc_type"],
    )
    val_docs = edsnlp.data.read_json(
        "./dataset/doc_types_dev.jsonl",
        converter="omop",
        doc_attributes=["doc_type"],
    )

    # 🎛️ OPTIMIZER
    optimizer = ScheduledOptimizer.draft(  # (2)!
        optim=torch.optim.AdamW,
        groups=[
            {
                "selector": "doc_classifier[.]embedding",
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
        ],
    )

    # 🚀 TRAIN
    train(
        nlp=nlp,
        train_data=TrainingData(
            data=train_docs,
            batch_size="8 docs",
            pipe_names=["doc_classifier"],
            shuffle="dataset",
        ),
        val_data=val_docs,
        scorer={"classif": metric},
        optimizer=optimizer,
        max_steps=400,
        validation_interval=100,
        grad_max_norm=1.0,
        num_workers=0,
        output_dir="artifacts",
        # cpu=True,  # (optional) use CPU instead of GPU/MPS
    )
    ```

    1. `eds.doc_pooler` supports `mean`, `max`, `sum`, `attention` and `cls`. `mean` is a solid
    default ; `attention` learns which words matter and usually helps on long notes ; `cls`
    reuses the transformer's own `[CLS]` vector, and therefore requires `eds.transformer` as the
    underlying embedding.
    2. What does "draft" mean here ? We'll let the train function pass the nlp object
    to the optimizer after it has been been `post_init`'ed : `post_init` is the operation that
    looks at some data, finds how many label the model must learn, and updates the model weights
    to have as many heads as there are labels observed in the train data. This function will be
    called by `train`, so the optimizer should be defined *after*, when the model parameter
    tensors are final. To do that, instead of instantiating the optimizer right now, we create
    a "Draft", which will be instantiated inside the `train` function, once all the required
    parameters are set.

That's the whole single-head story. Swapping `eds.single_label_head` for `eds.multi_label_head` is all it takes to move to an attribute that holds a *set* of labels — the topics a note covers, say — the gold column becoming a JSON list instead of a string.

## Several heads: coding a stay with its DP and DAS

French hospital stays are coded with exactly one **principal diagnosis** (*diagnostic principal*, DP) and a variable number of **associated diagnoses** (*diagnostics associés*, DAS), all ICD-10 codes. That's two different problems on the same note: a single-label one and a multi-label one. Rather than training two models, we give the classifier two heads over a **shared document embedding**, computed once.

The multi-label head decides on its own *how many* labels to predict: it keeps every label whose probability exceeds `threshold`, so a note with no relevant comorbidity gets an empty list, and one with three gets three. `threshold` is worth tuning on your dev set — lower it to favour recall, raise it to favour precision.

`coding_train.jsonl` therefore carries two label columns, a string for the DP and a list for the DAS:

```json { title="dataset/coding_train.jsonl" }
{"note_id": "201", "note_text": "COMPTE RENDU D'HOSPITALISATION — séjour du 23/04/2025 au 28/04/2025\nExacerbation aiguë d'une BPCO connue …\nAntécédents : Tabagisme sevré depuis deux ans, 25 paquets-années.", "dp": "J44.0", "das": ["F17.2"]}
{"note_id": "202", "note_text": "COMPTE RENDU D'HOSPITALISATION — séjour du 05/02/2025 au 12/02/2025\nColique hépatique fébrile …\nAntécédents : Pas d'antécédent notable.", "dp": "K80.2", "das": []}
```

=== "From the command line"

    ```yaml { title="configs/coding.yml" }
    vars:
      train: './dataset/coding_train.jsonl'
      dev: './dataset/coding_dev.jsonl'

    # 🤖 PIPELINE DEFINITION
    nlp:
      '@core': pipeline
      lang: eds

      components:
        coder:
          '@factory': eds.doc_classifier
          embedding:
            '@factory': eds.doc_pooler
            # Let the model learn which parts of the note to look at
            pooling_mode: 'attention'
            embedding:
              '@factory': eds.transformer
              model: 'almanach/camembert-bio-base'
              window: 128
              stride: 96

          heads:
            # The principal diagnosis: exactly one per stay
            dp:
              '@misc': eds.single_label_head
              loss: 'focal'  # (1)!
              hidden_size: 256
              dropout_rate: 0.1

            # The associated diagnoses: a set, of any size
            das:
              '@misc': eds.multi_label_head
              loss: 'bce'
              threshold: 0.5  # (2)!
              hidden_size: 256
              dropout_rate: 0.1

    # 📈 SCORER
    scorer:
      codes:
        '@metrics': eds.doc_classification
        label_attr: [ 'dp', 'das' ]

    # 🎛️ OPTIMIZER
    optimizer:
      "@core": optimizer !draft
      optim: torch.optim.AdamW
      groups:
        - selector: 'coder[.]embedding'
          lr:
            '@schedules': linear
            warmup_rate: 0.1
            start_value: 0.
            max_value: 5e-5
        - selector: '.*'
          lr:
            '@schedules': linear
            warmup_rate: 0.1
            start_value: 3e-4
            max_value: 3e-4

    # 📚 DATA
    train_data:
      - data:
          '@readers': json
          path: ${vars.train}
          converter:
            - '@factory': eds.omop_dict2doc
              doc_attributes: [ 'dp', 'das' ]
        shuffle: dataset
        batch_size: 8 docs
        pipe_names: [ "coder" ]

    val_data:
      '@readers': json
      path: ${vars.dev}
      converter:
        - '@factory': eds.omop_dict2doc
          doc_attributes: [ 'dp', 'das' ]

    # 🚀 TRAIN SCRIPT OPTIONS
    train:
      nlp: ${nlp}
      train_data: ${train_data}
      val_data: ${val_data}
      max_steps: 600
      validation_interval: 150
      max_grad_norm: 1.0
      scorer: ${scorer}
      num_workers: 1
      output_dir: 'artifacts'
    ```

    1. Diagnosis distributions have a very long tail. The focal loss down-weights the easy,
    frequent codes so that the rare ones keep contributing to the gradient. Class weights are
    available too, through the `class_weights` argument — a label → frequency mapping, or the
    path to a pickled one.
    2. The decision threshold above which a diagnosis is kept. Lower it to favour recall,
    raise it to favour precision — it costs nothing to re-tune on the dev set after training,
    since it only affects decoding.

    And train the model:

    ```bash { data-md-color-scheme="slate" }
    python -m edsnlp.train --config configs/coding.yml --seed 42
    ```

=== "From a script or a notebook"

    ```python { .no-check }
    import edsnlp
    import edsnlp.pipes as eds
    import torch
    from edsnlp.metrics.doc_classification import DocClassificationMetric
    from edsnlp.pipes.trainable.doc_classifier.heads import (
        MultiLabelHead,
        SingleLabelHead,
    )
    from edsnlp.training import ScheduledOptimizer, TrainingData, train

    # 🤖 PIPELINE DEFINITION
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.doc_classifier(
            embedding=eds.doc_pooler(
                # Let the model learn which parts of the note to look at
                pooling_mode="attention",
                embedding=eds.transformer(
                    model="almanach/camembert-bio-base",
                    window=128,
                    stride=96,
                ),
            ),
            heads={
                # The principal diagnosis: exactly one per stay
                "dp": SingleLabelHead(
                    loss="focal",  # (1)!
                    hidden_size=256,
                    dropout_rate=0.1,
                ),
                # The associated diagnoses: a set, of any size
                "das": MultiLabelHead(
                    loss="bce",
                    threshold=0.5,  # (2)!
                    hidden_size=256,
                    dropout_rate=0.1,
                ),
            },
        ),
        name="coder",
    )

    # 📈 SCORER
    metric = DocClassificationMetric(label_attr=["dp", "das"])

    # 📚 DATA
    train_docs = edsnlp.data.read_json(
        "./dataset/coding_train.jsonl",
        converter="omop",
        doc_attributes=["dp", "das"],
    )
    val_docs = edsnlp.data.read_json(
        "./dataset/coding_dev.jsonl",
        converter="omop",
        doc_attributes=["dp", "das"],
    )

    # 🎛️ OPTIMIZER
    optimizer = ScheduledOptimizer.draft(
        optim=torch.optim.AdamW,
        groups=[
            {
                "selector": "coder[.]embedding",
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
        ],
    )

    # 🚀 TRAIN
    train(
        nlp=nlp,
        train_data=TrainingData(
            data=train_docs,
            batch_size="8 docs",
            pipe_names=["coder"],
            shuffle="dataset",
        ),
        val_data=val_docs,
        scorer={"codes": metric},
        optimizer=optimizer,
        max_steps=600,
        validation_interval=150,
        grad_max_norm=1.0,
        num_workers=0,
        output_dir="artifacts",
        # cpu=True,  # (optional) use CPU instead of GPU/MPS
    )
    ```

    1. Diagnosis distributions have a very long tail. The focal loss down-weights the easy,
    frequent codes so that the rare ones keep contributing to the gradient. Class weights are
    available too, through the `class_weights` argument — a label → frequency mapping, or the
    path to a pickled one.
    2. The decision threshold above which a diagnosis is kept. Lower it to favour recall,
    raise it to favour precision — it costs nothing to re-tune on the dev set after training,
    since it only affects decoding.

!!! tip "Mixing partially annotated corpora"

    A head is only supervised by the documents that carry its gold attribute. This makes it easy
    to combine a small, fully annotated corpus with a larger one annotated for the principal
    diagnosis only: declare them as two training streams, and the extra notes will train the
    `dp` head (and the shared embedding) without ever touching the `das` one.

    ```python { .no-check }
    train(
        nlp=nlp,
        train_data=[
            TrainingData(
                data=train_docs,  # annotated with both dp and das
                batch_size="8 docs",
                pipe_names=["coder"],
                shuffle="dataset",
            ),
            TrainingData(
                data=edsnlp.data.read_json(
                    "./dataset/coding_dp_only.jsonl",
                    converter="omop",
                    doc_attributes=["dp"],  # nothing else is annotated
                ),
                batch_size="8 docs",
                pipe_names=["coder"],
                shuffle="dataset",
            ),
        ],
        ...
    )
    ```

    Declare them as two streams rather than concatenating them into one: a batch supervises a
    head only if *every* document in it carries the corresponding gold attribute.

## Use the model

You can now load the trained pipeline and apply it to new notes. Each head writes to the `Doc._` extension it is named after:

```python { .no-check }
import edsnlp

nlp = edsnlp.load("artifacts/model-last")

doc = nlp(
    "Admission pour douleur thoracique constrictive prolongée. "
    "La coronarographie retrouve une occlusion de l'artère interventriculaire "
    "antérieure, traitée par angioplastie. "
    "Antécédents : diabète de type 2, hypertension artérielle traitée."
)

doc._.dp   # (1)!
doc._.das  # (2)!
```

1. `'I21.9'` — a single code, since `dp` is a single-label head.
2. `['E11.9', 'I10']` — a list, holding every code whose probability passed the threshold.

To run the model over a whole corpus, use the [stream API](../concepts/inference.md):

```python { .no-check }
docs = edsnlp.data.read_parquet("./notes.parquet", converter="omop")
docs = docs.map_pipeline(nlp)
docs.write_parquet(
    "./predictions.parquet",
    converter="omop",
    doc_attributes=["dp", "das"],
)
```
