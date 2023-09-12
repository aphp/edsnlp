# Nested Named Entity Recognition

The default spaCy Named Entity Recognizer (NER) pipeline only allows flat entity
recognition, meaning that overlapping and nested entities cannot be extracted.

The other spaCy component `SpanCategorizer` only supports assigning to a specific span
group and both components are not well suited for extracting entities with ill-defined
boundaries (this can occur if your training data contains difficult and long entities).

We propose the new `eds.ner` component to extract almost any named entity:

- flat entities like spaCy's `EntityRecognizer` overlapping entities
- overlapping entities of different labels (much like spaCy's `SpanCategorizer`)
- entities will ill-defined boundaries

However, the model cannot currently extract entities that are nested inside larger entities
of the same label.

The pipeline assigns both `doc.ents` (in which overlapping entities are filtered
out) and `doc.spans`.

## Architecture

The model performs token classification using the BIOUL (Begin, Inside, Outside, Unary, Last) tagging scheme.
To extract overlapping entities, each label has its own tag sequence, so the model predicts
`n_labels` sequences of O, I, B, L, U tags. The architecture is displayed in the figure below.

To enforce the tagging scheme, (ex: I cannot follow O but only B, ...), we use a stack of
CRF (Conditional Random Fields) layers, one per label during both training and prediction.

<figure markdown>
  ![Nested NER architecture](./edsnlp-ner.svg)
  <figcaption>Nested NER architecture</figcaption>
</figure>

## Usage

Let us define the pipeline and train it:

```{ .python .no-check }
from pathlib import Path

import spacy

from edsnlp.connectors.brat import BratConnector
from edsnlp.utils.training import train, make_spacy_corpus_config

tmp_path = Path("/tmp/test-nested-ner")

nlp = spacy.blank("eds")
# ↓ below is the nested ner pipeline ↓
# you can configure it using the `add_pipe(..., config=...)` parameter
nlp.add_pipe("nested_ner")

# Train the model, with additional training configuration
nlp = train(
    nlp,
    output_path=tmp_path / "model",
    config=dict(
        **make_spacy_corpus_config(
            train_data="/path/to/the/training/set/brat/files",
            dev_data="/path/to/the/dev/set/brat/files",
            nlp=nlp,
            data_format="brat",
        ),
        training=dict(
            max_steps=4000,
        ),
    ),
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

## Configuration

The pipeline component can be configured using the following parameters :

<div markdown="1" class="explicit-col-width">

::: edsnlp.pipelines.trainable.nested_ner.factory.create_component
    options:
      only_parameters: true

The default model `eds.nested_ner_model.v1` can be configured using the following parameters :

::: edsnlp.pipelines.trainable.nested_ner.stack_crf_ner.create_model
    options:
      only_parameters: true

</div>

## Authors and citation

The `eds.nested_ner` pipeline was developed by AP-HP's Data Science team.

The deep learning model was adapted from Wajsbürt[@wajsburt:tel-03624928]

\bibliography
