# Trainable Document Classifier {: #edsnlp.pipes.trainable.doc_classifier.factory.create_component }

::: edsnlp.pipes.trainable.doc_classifier.factory.create_component
    options:
        heading_level: 2
        show_bases: false
        show_source: false
        only_class_level: true

## Heads

The classifier is configured with one head per predicted attribute. Pick
`eds.single_label_head` when a document carries exactly one label for that
attribute, and `eds.multi_label_head` when it carries a set of them. Both accept
the [parameters common to every head][edsnlp.pipes.trainable.doc_classifier.heads.ClassificationHead]
— label set, hidden block, class weights — on top of their own.

### Single-label head {: #edsnlp.pipes.trainable.doc_classifier.heads.SingleLabelHead }

::: edsnlp.pipes.trainable.doc_classifier.heads.SingleLabelHead
    options:
        heading_level: 3
        show_bases: false
        show_source: false
        only_class_level: true

### Multi-label head {: #edsnlp.pipes.trainable.doc_classifier.heads.MultiLabelHead }

::: edsnlp.pipes.trainable.doc_classifier.heads.MultiLabelHead
    options:
        heading_level: 3
        show_bases: false
        show_source: false
        only_class_level: true

### Parameters common to every head {: #edsnlp.pipes.trainable.doc_classifier.heads.ClassificationHead }

::: edsnlp.pipes.trainable.doc_classifier.heads.ClassificationHead
    options:
        heading_level: 3
        show_bases: false
        show_source: false
        only_class_level: true
