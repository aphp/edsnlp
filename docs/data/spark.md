# Spark

??? abstract "TLDR"

    ```{ .python .no-check }
    import edsnlp

    iterator = edsnlp.data.from_spark(df, converter="omop")
    docs = nlp.pipe(iterator)
    res = edsnlp.data.to_spark(docs, converter="omop", dtypes=None) # dtypes (i.e. pyspark's schema) is recommended
    ```

We provide methods to read and write documents (raw or annotated) from and to Spark DataFrames.

As an example, imagine that we have the following OMOP dataframe (we'll name it `note_df`)

| note_id | note_text                                     | note_datetime |
|--------:|:----------------------------------------------|:--------------|
|       0 | Le patient est admis pour une pneumopathie... | 2021-10-23    |

## Reading from a Spark Dataframe {: #edsnlp.data.spark.SparkReader }

::: edsnlp.data.spark.SparkReader
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false

## Writing to a Spark DataFrame {: #edsnlp.data.spark.to_spark }

::: edsnlp.data.spark.to_spark
    options:
        heading_level: 3
        show_source: false
        show_toc: false
        show_bases: false

## Importing entities from a Spark DataFrame

If you have a dataframe with entities (e.g., `note_nlp` in OMOP), you must join it with the dataframe containing the raw text (e.g., `note` in OMOP) to obtain a single dataframe with the entities next to the raw text. For instance, the second `note_nlp` dataframe that we will name `note_nlp`.

| note_nlp_id | note_id | start_char | end_char | note_nlp_source_value | lexical_variant |
|------------:|--------:|-----------:|---------:|:----------------------|:----------------|
|           0 |       0 |         46 |       57 | disease               | coronavirus     |
|           1 |       0 |         77 |       88 | drug                  | paracétamol     |

```{ .python .no-check }
import pyspark.sql.functions as F

df = note_df.join(
    note_nlp_df
    .groupBy("note_id")
    .agg(
        F.collect_list(
            F.struct(
                F.col("note_nlp_id"),
                F.col("start_char"),
                F.col("end_char"),
                F.col("note_nlp_source_value")
            )
        ).alias("entities")
    ), "note_id", "left")
```
