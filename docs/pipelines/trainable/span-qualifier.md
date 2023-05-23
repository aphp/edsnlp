# Span qualification

We propose the new `span_qualifier` component to qualify (i.e. assign attributes on) any span with machine learning.
In this context, the span qualification task consists in assigning values (boolean, strings or any complex object) to attributes/extensions of spans such as:

- `span.label_`,
- `span._.negation`,
- `span._.date.mode`
- etc.

## Architecture

The underlying `eds.span_multilabel_classifier.v1` model performs span classification by:

1. Pooling the words embedding (`mean`, `max` or `sum`) into a single embedding per span
2. Computing logits for each possible binding (i.e. qualifier-value assignment)
3. Splitting these bindings into independent groups such as

    - `event_type=start` and `event_type=stop`
    - `negated=False` and `negated=True`

4. Learning or predicting a combination amongst legal combination of these bindings.
For instance in the second group, we can't have both `negated=True` and `negated=False` so the combinations are `[(1, 0), (0, 1)]`
5. Assigning bindings on spans depending on the predicted results

## Usage

Let us define the pipeline and train it. We provide utils to train the model using an API, but you can use a spaCy's config file as well.


=== "API-based"

    <!-- no-check -->

    ```python
    from pathlib import Path

    import spacy

    from edsnlp.connectors.brat import BratConnector
    from edsnlp.utils.training import train, make_spacy_corpus_config
    from edsnlp.pipelines.trainable.span_qualifier import SPAN_QUALIFIER_DEFAULTS

    tmp_path = Path("/tmp/test-span-qualifier")

    nlp = spacy.blank("eds")
    # ↓ below is the span qualifier pipeline ↓
    # you can configure it using the `add_pipe(..., config=...)` parameter
    nlp.add_pipe(
        "span_qualifier",
        config={
            **SPAN_QUALIFIER_DEFAULTS,
            # Two qualifiers: binary `_.negation` and multi-class `_.event_type`
            "qualifiers": ("_.negation", "_.event_type"),
            # Only predict on entities, not on span groups
            "from_ents": True,
            "from_span_groups": False,
            "label_constraints": {
                # Only allow `_.event_type` qualifier on events
                "_.event_type": ("event",),
            },
            "model": {
                **SPAN_QUALIFIER_DEFAULTS["model"],
                "pooler_mode": "mean",
                "classifier_mode": "dot",
            },
        },
    )

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
                max_steps=100,
            ),
        ),
    )

    # Finally, we can run the pipeline on a new document
    doc = nlp.make_doc("Arret du ttt si folfox inefficace")
    doc.ents = [
        # event = "Arret"
        spacy.tokens.Span(doc, 0, 1, "event"),
        # criteria = "si"
        spacy.tokens.Span(doc, 3, 4, "criteria"),
        # drug = "folfox"
        spacy.tokens.Span(doc, 4, 5, "drug"),
    ]
    [ent._.negation for ent in doc.ents]
    # Out: [True, False, False]

    [ent._.event_type for ent in doc.ents]
    # Out: ["start", None, None]

    # And export new predictions as Brat annotations
    predicted_docs = BratConnector("/path/to/the/new/files", run_pipe=True).brat2docs(nlp)
    BratConnector("/path/to/predictions").docs2brat(predicted_docs)
    ```

=== "Configuration-based"

    ```ini title="config.cfg"

    [paths]
    train = null
    dev = null
    vectors = null
    init_tok2vec = null
    raw = null

    [system]
    seed = 0
    gpu_allocator = null

    [nlp]
    lang = "eds"
    pipeline = ["span_qualifier"]

    [components]

    [components.span_qualifier]
    factory = "span_qualifier"
    label_constraints = null
    from_ents = false
    from_span_groups = true
    qualifiers = ["label_"]
    scorer = {"@scorers":"eds.span_qualifier_scorer.v1"}

    [components.span_qualifier.model]
    @architectures = "eds.span_multi_classifier.v1"
    projection_mode = "dot"
    pooler_mode = "max"
    n_labels = null

    [components.span_qualifier.model.tok2vec]
    @architectures = "spacy.Tok2Vec.v1"

    [components.span_qualifier.model.tok2vec.embed]
    @architectures = "spacy.MultiHashEmbed.v1"
    width = 96
    rows = [5000,2000,1000,1000]
    attrs = ["ORTH","PREFIX","SUFFIX","SHAPE"]
    include_static_vectors = false

    [components.span_qualifier.model.tok2vec.encode]
    @architectures = "spacy.MaxoutWindowEncoder.v1"
    width = 96
    window_size = 1
    maxout_pieces = 3
    depth = 4

    [corpora]

    [corpora.train]
    @readers = "test-span-classification-corpus"
    path = ${path.train}
    max_length = 0
    gold_preproc = false
    limit = 0
    augmenter = null

    [corpora.dev]
    @readers = "test-span-classification-corpus"
    path = ${path.dev}
    max_length = 0
    gold_preproc = false
    limit = 0
    augmenter = null

    [training]
    seed = ${system.seed}
    gpu_allocator = ${system.gpu_allocator}
    dropout = 0.1
    accumulate_gradient = 1
    patience = 10000
    max_epochs = 0
    max_steps = 10
    eval_frequency = 5
    frozen_components = []
    annotating_components = []
    dev_corpus = "corpora.dev"
    train_corpus = "corpora.train"
    before_to_disk = null
    before_update = null

    [training.batcher]
    @batchers = "spacy.batch_by_words.v1"
    discard_oversize = false
    tolerance = 0.2
    get_length = null

    [training.batcher.size]
    @schedules = "compounding.v1"
    start = 100
    stop = 1000
    compound = 1.001
    t = 0.0

    [training.logger]
    @loggers = "spacy.ConsoleLogger.v1"
    progress_bar = false

    [training.optimizer]
    @optimizers = "Adam.v1"
    beta1 = 0.9
    beta2 = 0.999
    L2_is_weight_decay = true
    L2 = 0.01
    grad_clip = 1.0
    use_averages = false
    eps = 0.00000001
    learn_rate = 0.001

    [training.score_weights]
    accuracy = 1.0

    [pretraining]

    [initialize]
    vectors = ${paths.vectors}
    init_tok2vec = ${paths.init_tok2vec}
    vocab_data = null
    lookups = null
    before_init = null
    after_init = null

    [initialize.components]

    [initialize.tokenizer]

    ```

    ```bash
    spacy train config.cfg --output training/ --paths.train your_corpus/train.spacy --paths.dev your_corpus/dev.spacy
    ```

## Configuration

The `span_qualifier` pipeline component can be configured using the following parameters :

::: edsnlp.pipelines.trainable.span_qualifier.factory.create_component
    options:
      only_parameters: true

The default model `eds.span_multi_classifier.v1` can be configured using the following parameters :

::: edsnlp.pipelines.trainable.span_qualifier.span_multi_classifier.create_model
    options:
      only_parameters: true

## Authors and citation

The `eds.span_qualifier` pipeline was developed by AP-HP's Data Science team.

\bibliography
