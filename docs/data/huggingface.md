# HuggingFace datasets

??? abstract "TLDR"

    ```{ .python .no-check }
    import edsnlp

    # Read from the Hub (streaming) and convert to Docs
    stream = edsnlp.data.from_huggingface_dataset(
        "lhoestq/conll2003",
        split="train",
        converter="hf_ner",
        tag_order=[
            "O",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "B-MISC",
            "I-MISC",
        ],
        nlp=edsnlp.blank("eds"),
        load_kwargs={"streaming": True},
    )

    # Optionally process
    stream = stream.map_pipeline(nlp)

    # Export back to a HF IterableDataset of dicts
    hf_iter = edsnlp.data.to_huggingface_dataset(
        stream,
        converter="hf_ner",
        words_column="tokens",
        ner_tags_column="ner_tags",
    )
    ```

Use the Hugging Face Datasets ecosystem as a data source or sink for EDS-NLP pipelines. You can read datasets from the Hub or reuse already loaded `datasets.Dataset` / `datasets.IterableDataset` objects, optionally shuffle them deterministically, loop over them, and map them through any pipeline before writing them back as an `IterableDataset`.

We rely on the `datasets` package. Install it with `pip install datasets` or `pip install "edsnlp[ml]"`.

Typical converters:

- `hf_ner`: expects token and tag columns (defaults: `tokens`, `ner_tags`) and produces Docs with entities. Compatible with BILOU/IOB schemes through `tag_order` or `tag_map`.
- `hf_text`: expects a single text column (default: `text`) and produces plain Docs; optional `id_column` is inferred when present.

When loading a dataset dictionary with multiple splits, pass an explicit `split` (e.g. `"train"`). You can also select a configuration/subset via `name` and forward any `datasets.load_dataset` arguments through `load_kwargs` (e.g. `{"streaming": True}`).

## Reading Hugging Face datasets {: #edsnlp.data.huggingface_dataset.from_huggingface_dataset }

::: edsnlp.data.huggingface_dataset.from_huggingface_dataset
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false

## Writing Hugging Face datasets {: #edsnlp.data.huggingface_dataset.to_huggingface_dataset }

::: edsnlp.data.huggingface_dataset.to_huggingface_dataset
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false
