# Inference

Once you have obtained a pipeline, either by composing rule-based components, training a model or loading a model from the disk, you can use it to make predictions on documents. This is referred to as inference. This page answers the following questions :

> How do we leverage computational resources run a model on many documents?

> How do we connect to various data sources to retrieve documents?

## Inference on a single document

In EDS-NLP, computing the prediction on a single document is done by calling the pipeline on the document. The input can be either:

- a text string
- or a [Doc](https://spacy.io/api/doc) object

```{ .python .no-check }
from pathlib import Path

nlp = ...
text = "... my text ..."
doc = nlp(text)
```

If you're lucky enough to have a GPU, you can use it to speed up inference by moving the model to the GPU before calling the pipeline. To leverage multiple GPUs, refer to the [multiprocessing accelerator][edsnlp.accelerators.multiprocessing.MultiprocessingAccelerator] description below.

```{ .python .no-check }
nlp.to("cuda")  # same semantics as pytorch
doc = nlp(text)
```

## Inference on multiple documents

When processing multiple documents, it is usually more efficient to use the `nlp.pipe(...)` method, especially when using deep learning components, since this allow matrix multiplications to be batched together. Depending on your computational resources and requirements, EDS-NLP comes with various "accelerators" to speed up inference (see the [Accelerators](#accelerators) section for more details). By default, the `.pipe()` method uses the [`simple` accelerator][edsnlp.accelerators.simple.SimpleAccelerator] but you can switch to a different one by passing the `accelerator` argument.

```{ .python .no-check }
nlp = ...
docs = nlp.pipe(
    [text1, text2, ...],
    batch_size=16,  # optional, default to the one defined in the pipeline
    accelerator=my_accelerator,
)
```

The `pipe` method supports the following arguments :

::: edsnlp.core.pipeline.Pipeline.pipe
    options:
        heading_level: 3
        only_parameters: true

## Accelerators

### Simple accelerator {: #edsnlp.accelerators.simple.SimpleAccelerator }

::: edsnlp.accelerators.simple.SimpleAccelerator
    options:
        heading_level: 3
        only_class_level: true

### Multiprocessing (GPU) accelerator {: #edsnlp.accelerators.multiprocessing.MultiprocessingAccelerator }

::: edsnlp.accelerators.multiprocessing.MultiprocessingAccelerator
    options:
        heading_level: 3
        only_class_level: true
