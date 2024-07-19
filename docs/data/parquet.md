# Parquet

??? abstract "TLDR"

    ```{ .python .no-check }
    import edsnlp

    docs = edsnlp.data.read_parquet(df, converter="omop")
    docs = docs.map_pipeline(nlp)
    res = edsnlp.data.to_parquet(docs, converter="omop")
    ```

We provide methods to read and write documents (raw or annotated) from and to parquet files.

As an example, imagine that we have the following document that uses the OMOP schema (parquet files are not actually stored as human-readable text, but this is for the sake of the example):

```{ title="data.pq" }
{ "note_id": 0, "note_text": "Le patient ...", "note_datetime": "2021-10-23", "entities": [...] }
{ "note_id": 1, "note_text": "Autre doc ...", "note_datetime": "2022-12-24", "entities": [] }
...
```

You could also have multiple parquet files in a directory, the reader will read them all.

## Reading Parquet files {: #edsnlp.data.parquet.read_parquet }

::: edsnlp.data.parquet.read_parquet
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false

## Writing Parquet files {: #edsnlp.data.parquet.write_parquet }

::: edsnlp.data.parquet.write_parquet
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false
