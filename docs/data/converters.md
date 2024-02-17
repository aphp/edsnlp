# Schemas {: #data-schemas }

Data can be read from and writen to various sources, like JSON/BRAT/CSV files or dataframes, and arranged following
different schemas. The schema defines the structure of the data and how it should be interpreted. We detail
here the different schemas that are currently supported, and how to configure them to be converted from and to Doc objects.

These parameters can be passed to the `from_*` (or `read_*` in the case of files) and `to_*` (or `write_*` in the case of files) methods, depending on the chosen `converter` argument.

## OMOP

OMOP is a schema that is used in the medical domain. It is based on the [OMOP Common Data Model](https://ohdsi.github.io/CommonDataModel/). We are mainly interested in the `note` table, which contains
the clinical notes, and deviate from the original schema by adding an *optional* `entities` column that can be computed from the `note_nlp` table.

Therefore, a complete OMOP-style document would look like this:

```{ .json }
{
  "note_id": 0,
  "note_text": "Le patient ...",
  "entities": [
    {
      "note_nlp_id": 0,
      "start_char": 3,
      "end_char": 10,
      "lexical_variant": "patient",
      "note_nlp_source_value": "person",

      # optional fields
      "negated": False,
      "certainty": "probable",
      ...
    },
    ...
  ],

  # optional fields
  "custom_doc_field": "..."
  ...
}
```

### Converting OMOP data to Doc objects

::: edsnlp.data.converters.OmopDict2DocConverter
    options:
        heading_level: 3
        only_parameters: "no-header"

### Converting Doc objects to OMOP data

::: edsnlp.data.converters.OmopDoc2DictConverter
    options:
        heading_level: 3
        only_parameters: "no-header"

## Standoff

Standoff refers mostly to the [BRAT standoff format](https://brat.nlplab.org/standoff.html), but doesn't indicate how
the annotations should be stored in a JSON-like schema. We use the following schema:

```{ .json }
{
  "doc_id": 0,
  "text": "Le patient ...",
  "entities": [
    {
      "entity_id": 0,
      "label": "drug",
      "fragments": [{
        "start": 0,
        "end": 10
      }],
      "attributes": {
        "negated": True,
        "certainty": "probable"
      }
    },
    ...
  ]
}
```

### Converting Standoff data to Doc objects

::: edsnlp.data.converters.StandoffDict2DocConverter
    options:
        heading_level: 3
        only_parameters: "no-header"

### Converting Doc objects to Standoff data

::: edsnlp.data.converters.StandoffDoc2DictConverter
    options:
        heading_level: 3
        only_parameters: "no-header"
