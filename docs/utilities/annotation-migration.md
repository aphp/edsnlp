# Migrating annotations between document versions

`edsnlp.utils.fuzzy_alignment.align` realigns entity offsets when you need to
transfer BRAT annotations from an older release of a document to a slightly
edited version. The function works directly on the raw standoff dictionaries
produced by `edsnlp.data.read_standoff`, so there is no need to materialise
SpaCy `Doc` objects.

## Load the BRAT exports as dictionaries

??? tip "Create fake directories for testing"

    For demonstration purposes, let's create two sample BRAT directories with
    minimal content.

    ```python
    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.covid")
    nlp.add_pipe("eds.diabetes")

    # Annotated documents
    data = edsnlp.data.from_iterable(
        [
            {
                "note_id": "doc_1",
                "note_text": "Le diabete, c'est terrible. La patient a le covid, il est malade.",
            }
        ],
        converter="omop",
    )
    data.map_pipeline(nlp).write_standoff("test_data/v1/", overwrite=True)

    # Slightly modified documents, with no annotations
    # (ie we don't run the pipeline here)
    data = edsnlp.data.from_iterable(
        [
            {
                "note_id": "doc_1",
                "note_text": "Le patient a le CO-VID, il est malade. Le diabete, c'est terrible.",
            }
        ],
        converter="omop",
    )
    data.write_standoff("test_data/v2/", overwrite=True)
    ```

Let's read two BRAT exports as dictionaries: one from the old version of the
documents, which are annotated, and one from the new version which only contains
the updated text.

```python
import edsnlp

old_docs = {
    doc["doc_id"]: doc
    for doc in edsnlp.data.read_standoff("test_data/v1", converter=None)
}
new_docs = {
    doc["doc_id"]: doc
    for doc in edsnlp.data.read_standoff(
        "test_data/v2", converter=None, keep_txt_only_docs=True
    )
}
```

Each entry is a plain dict with the following structure:

```python
{
    "doc_id": "note-123",
    "text": "...raw document text...",
    "entities": [
        {
            "label": "PROBLEM",
            "fragments": [{"begin": 128, "end": 143}],
        },
        # ...
    ],
}
```

The updated documents expose the text while leaving `entities` empty, which is
exactly what the alignment utility expects.

## Align entity spans on the new text

```python
from edsnlp.utils.fuzzy_alignment import align


def migrate_entities(doc_id, old_doc, new_doc, threshold=0.75, verbose=0):
    result = align(
        old=old_doc,
        new=new_doc,
        threshold=threshold,
        do_debug=verbose >= 2,  # switch to True to inspect ambiguous matches
    )
    if verbose >= 1:
        print(
            f"{doc_id}: {result['good_count']} migrated, "
            f"{result['unsure_count']} unsure, "
            f"{result['missing_count']} missing"
        )
        if result.get("good"):
            print(f"\n{doc_id} - entities that should be ok:")
            text = old_doc["text"]
            for fragment in result["good"]:
                print(" -", text[fragment["begin"] : fragment["end"]])
        if result.get("unsure"):
            print(f"\n{doc_id} - entities that require reviewing:")
            text = old_doc["text"]
            for fragment in result["unsure"]:
                print(" -", text[fragment["begin"] : fragment["end"]])
        if result.get("missing"):
            print(f"\n{doc_id} - entities that couldn't be found:")
            text = old_doc["text"]
            for fragment in result["missing"]:
                print(" -", text[fragment["begin"] : fragment["end"]])
    return result


migration_results = {}
migrated_docs = []

for doc_id, old_doc in old_docs.items():
    try:
        new_doc = new_docs[doc_id]
    except KeyError:
        print(f"Skipping {doc_id}: no matching new text found")
        continue

    result = migrate_entities(doc_id, old_doc, new_doc, verbose=1)
    migration_results[doc_id] = result
    migrated_docs.append(result["doc"])
```

which prints:
```bash { data-md-color-scheme="slate" }
doc_1: 2 migrated, 0 unsure, 0 missing

doc_1 - entities that should be ok:
 - diabete
 - covid
```

The helper returns:

- `doc`: the new standoff dictionary with entity offsets projected on the
  updated text.
- `missing`, `good`, `unsure`: fragment lists that help audit the migration.

Tune the `threshold` or the `sim_scheme` argument if many fragments land in the
`unsure` bucket. Setting `do_debug=True` prints the surrounding context for each
candidate span, which is handy when adjusting similarities.


Before exporting, you can print a a report of the fragments that could
not be matched confidently by passing `verbose=1` to `migrate_entities`.

These remaining spans can be handed back to annotators for manual validation.

## Export the migrated annotations

```python
output_dir = "data/brat/v2-migrated"
edsnlp.datawrite_standoff(
    migrated_docs,
    output_dir,
    overwrite=False,
    converter=None,  # keep the dictionaries as-is
)
```

The exporter re-creates `.txt` and `.ann` pairs compatible with BRAT. Because
we are passing dictionaries directly, you must keep `converter=None` when
calling both `read_standoff` and `write_standoff`.

At this stage you can open the new directory in BRAT, spot-check the migrated
documents, and iterate on alignment settings if needed.
