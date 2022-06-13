# Trainable components overview

In addition to its rule-based pipeline components, EDS-NLP offers new trainable pipelines to fit and run machine learning models for classic information extraction tasks.

## Available components :

| Name                  | Description                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| `eds.nested_ner`      | Recognize overlapping or nested entities (replaces spacy's `ner` component) |

!!! note "Writing custom models"

    Spacy models can be written with Thinc (Spacy's deep learning library), Tensorflow or Pytorch. As Pytorch is predominant in the NLP research field, we recommend writing models with the latter to facilitate interactions with the NLP community.

## Utils

### Training

In addition to the spacy `train` CLI, EDS-NLP offers a `train` function that can be called in Python directly with an existing spacy pipeline.

### Logging

EDS-NLP also offers a pretty table logger `eds.RichLogger.v1` built on rich which acts as a replacement for `spacy.ConsoleLogger.v1`.

## Usage

Let us define and train a full pipeline :

<!-- no-check -->
```python
from pathlib import Path

import spacy
from spacy.tokens import DocBin

from edsnlp.connectors.brat import BratConnector
from edsnlp.utils.training import train

tmp_path = Path("/tmp/test-train")

nlp = spacy.blank("eds")
nlp.add_pipe("nested_ner")  # (1)

# We then convert our dataset to spacy format
DocBin(
    docs=BratConnector(
        "/path/to/the/training/set/brat/files",
    ).brat2docs(nlp)
).to_disk(tmp_path / "train.spacy")
DocBin(
    docs=BratConnector(
        "/path/to/the/dev/set/brat/files",
    ).brat2docs(nlp)
).to_disk(tmp_path / "dev.spacy")

# And train the model, with additional training configuration
nlp = train(
    nlp,
    output_path=tmp_path / "model",
    config={
        "paths": {
            "train": str(tmp_path / "train.spacy"),
            "dev": str(tmp_path / "dev.spacy"),
        },
        "training": {
            "max_steps": 4000,
        },
    },
)

# Finally, we can run the pipeline on a new document
doc = nlp("Arret du folfox si inefficace")
doc.spans["drug"]
# Out: [folfox]

doc.spans["criteria"]
# Out: [si folfox inefficace]

# And export new predictions as Brat annotations
predicted_docs = BratConnector("/path/to/the/new/files", run_pipe=True).brat2docs(nlp)
BratConnector("/path/to/predictions").docs2brat(predicted_docs)
```

1. you can configure the component using the `add_pipe(..., config=...)` parameter
