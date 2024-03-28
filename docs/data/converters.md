# Converters {: #converters }

Data can be read from and writen to various sources, like JSON/BRAT/CSV files or dataframes, which expect a key-value representation and not Doc object.
For that purpose, we document here a set of converters that can be used to convert between these representations and Doc objects.

Converters can be configured in the `from_*` (or `read_*` in the case of files) and `to_*` (or `write_*` in the case of files) methods, depending on the chosen `converter` argument, which can be:

- a function, in which case it will be interpreted as a custom converter
- a string, in which case it will be interpreted as the name of a pre-defined converter

## Custom converter {: #custom }

You can always define your own converter functions to convert between your data and Doc objects.

### Reading from a custom schema

```{ .python .no-check }
import edsnlp, edsnlp.pipes as eds
from spacy.tokens import Doc
from edsnlp.data.converters import get_current_tokenizer
from typing import Dict

def convert_row_to_dict(row: Dict) -> Doc:
    # Tokenizer will be inferred from the pipeline
    doc = get_current_tokenizer()(row["custom_content"])
    doc._.note_id = row["custom_id"]
    doc._.note_datetime = row["custom_datetime"]
    # ...
    return doc

nlp = edsnlp.blank("eds")
nlp.add_pipe(eds.normalizer())
nlp.add_pipe(eds.covid())

# Any kind of reader (`edsnlp.data.read/from_...`) can be used here
docs = edsnlp.data.from_pandas(
    # Path to the file or directory
    dataframe,
    # How to convert JSON-like samples to Doc objects
    converter=convert_row_to_dict,
)
docs = docs.map_pipeline(nlp)
```

### Writing to a custom schema

```{ .python .no-check }
def convert_doc_to_row(doc: Doc) -> Dict:
    return {
        "custom_id": doc._.id,
        "custom_content": doc.text,
        "custom_datetime": doc._.note_datetime,
        # ...
    }

# Any kind of writer (`edsnlp.data.write/to_...`) can be used here
docs.write_parquet(
    "path/to/output_folder",
    # How to convert Doc objects to JSON-like samples
    converter=convert_doc_to_row,
)
```

!!! note "One row per entity"

    This function can also return a list of dicts, for instance one dict per detected entity, that will be treated as multiple rows in dataframe writers (e.g., `to_pandas`, `to_spark`, `write_parquet`).

    ```{ .python .no-check }
    def convert_ents_to_rows(doc: Doc) -> List[Dict]:
        return [
            {
                "note_id": doc._.id,
                "ent_text": ent.text,
                "ent_label": ent.label_,
                "custom_datetime": doc._.note_datetime,
                # ...
            }
            for ent in doc.ents
        ]


    docs.write_parquet(
        "path/to/output_folder",
        # How to convert entities of Doc objects to JSON-like samples
        converter=convert_ents_to_rows,
    )
    ```

## OMOP (`converter="omop"`) {: #omop }

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
        heading_level: 4
        show_source: false

### Converting Doc objects to OMOP data

::: edsnlp.data.converters.OmopDoc2DictConverter
    options:
        heading_level: 4
        show_source: false

## Standoff (`converter="standoff"`) {: #standoff }

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
        heading_level: 4
        show_source: false

### Converting Doc objects to Standoff data

::: edsnlp.data.converters.StandoffDoc2DictConverter
    options:
        heading_level: 4
        show_source: false



## Entities (`converter="ents"`) {: #ents }

We also provide a simple one-way (export) converter to convert Doc into a list of dictionaries,
one per entity, that can be used to write to a dataframe. The schema of each produced row is the following:

```{ .json }
{
    "note_id": 0,
    "start": 3,
    "end": 10,
    "label": "drug",
    "lexical_variant": "patient",

    # Optional fields
    "negated": False,
    "certainty": "probable"
    ...
}
```

::: edsnlp.data.converters.EntsDoc2DictConverter
    options:
        heading_level: 4
        show_source: false
