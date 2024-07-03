# Trainable components overview

In addition to its rule-based pipeline components, EDS-NLP offers new trainable components to fit and run machine learning models for classic biomedical information extraction tasks.

All trainable components implement the [`TorchComponent`][edsnlp.core.torch_component.TorchComponent] class, which provides a common API for training and inference.

## Available components :

<!-- --8<-- [start:components] -->

| Name                  | Description                                                           |
|-----------------------|-----------------------------------------------------------------------|
| `eds.transformer`     | Embed text with a transformer model                                   |
| `eds.text_cnn`        | Contextualize embeddings with a CNN                                   |
| `eds.span_pooler`     | A span embedding component that aggregates word embeddings            |
| `eds.ner_crf`         | A trainable component to extract entities                             |
| `eds.span_classifier` | A trainable component for multi-class multi-label span classification |
| `eds.span_linker`     | A trainable entity linker (i.e. to a list of concepts)                |

<!-- --8<-- [end:components] -->
