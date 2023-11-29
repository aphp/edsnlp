# JSON

??? abstract "TLDR"

    ```{ .python .no-check }
    import edsnlp

    iterator = edsnlp.data.from_pandas(df, converter="omop")
    docs = nlp.pipe(iterator)
    res = edsnlp.data.to_pandas(docs, converter="omop")
    ```

We provide methods to read and write documents (raw or annotated) from and to json files.

As an example, imagine that we have the following document that uses the OMOP schema

```{ title="data.jsonl" }
{ "note_id": 0, "note_text": "Le patient ...", "note_datetime": "2021-10-23", "entities": [...] }
{ "note_id": 1, "note_text": "Autre doc ...", "note_datetime": "2022-12-24", "entities": [] }
...
```

You could also have multiple `.json` files in a directory, the reader will read them all.

## Reading JSON files {: #edsnlp.data.json.read_json }

::: edsnlp.data.json.read_json
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false

## Writing JSON files {: #edsnlp.data.json.write_json }

::: edsnlp.data.json.write_json
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false
