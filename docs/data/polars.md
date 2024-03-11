# Polars

??? abstract "TLDR"

    ```{ .python .no-check }
    import edsnlp

    docs = edsnlp.data.from_polars(df, converter="omop")
    docs = docs.map_pipeline(nlp)
    res = edsnlp.data.to_polars(docs, converter="omop")
    ```

We provide methods to read and write documents (raw or annotated) from and to Polars DataFrames.

As an example, imagine that we have the following OMOP dataframe (we'll name it `note_df`)

| note_id | note_text                                     | note_datetime |
|--------:|:----------------------------------------------|:--------------|
|       0 | Le patient est admis pour une pneumopathie... | 2021-10-23    |

## Reading from a Polars Dataframe {: #edsnlp.data.polars.from_polars }

::: edsnlp.data.polars.from_polars
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false


## Writing to a Polars DataFrame {: #edsnlp.data.polars.to_polars }

::: edsnlp.data.polars.to_polars
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false
