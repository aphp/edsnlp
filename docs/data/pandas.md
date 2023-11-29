# Pandas

??? abstract "TLDR"

    ```{ .python .no-check }
    import edsnlp

    iterator = edsnlp.data.from_pandas(df, converter="omop")
    docs = nlp.pipe(iterator)
    res = edsnlp.data.to_pandas(docs, converter="omop")
    ```

We provide methods to read and write documents (raw or annotated) from and to Pandas DataFrames.

As an example, imagine that we have the following OMOP dataframe (we'll name it `note_df`)

| note_id | note_text                                     | note_datetime |
|--------:|:----------------------------------------------|:--------------|
|       0 | Le patient est admis pour une pneumopathie... | 2021-10-23    |

## Reading from a Pandas Dataframe {: #edsnlp.data.pandas.from_pandas }

::: edsnlp.data.pandas.from_pandas
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false


## Writing to a Pandas DataFrame {: #edsnlp.data.pandas.to_pandas }

::: edsnlp.data.pandas.to_pandas
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false


## Importing entities from a Pandas DataFrame

If you have a dataframe with entities (e.g., `note_nlp` in OMOP), you must join it with the dataframe containing the raw text (e.g., `note` in OMOP) to obtain a single dataframe with the entities next to the raw text. For instance, the second `note_nlp` dataframe that we will name `note_nlp_df`.

| note_nlp_id | note_id | start_char | end_char | note_nlp_source_value | lexical_variant |
|------------:|--------:|-----------:|---------:|:----------------------|:----------------|
|           0 |       0 |         46 |       57 | disease               | coronavirus     |
|           1 |       0 |         77 |       88 | drug                  | paracétamol     |
|         ... |     ... |        ... |      ... | ...                   | ...             |

```{ .python .no-check }
df = (
    note_df
    .set_index("note_id")
    .join(
        note_nlp_df
        .set_index('note_id')
        .groupby(level=0)
        .apply(pd.DataFrame.to_dict, orient='records')
        .rename("entities")
    )
).reset_index()
```

| note_id | note_text     | note_datetime |                                     entities |
|--------:|---------------|---------------|---------------------------------------------:|
|       0 | Le patient... | 2021-10-23    | `[{"note_nlp_id": 0, "start_char": 46, ...]` |
|     ... | ...           | ...           |                                          ... |
