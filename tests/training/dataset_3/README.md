# `dataset_3` — document-level classification

A small **synthetic** corpus used by the
[Training a document classifier](https://aphp.github.io/edsnlp/latest/tutorials/training-doc-classifier/)
tutorial. The notes were generated from a handful of templates: they contain no
real patient data, and the scores obtained on them are not meaningful.

| File | Documents | Annotated with |
|---|---|---|
| `doc_types_train.jsonl` | 60 | `doc_type` |
| `doc_types_dev.jsonl` | 16 | `doc_type` |
| `coding_train.jsonl` | 60 | `dp`, `das`, `das_count` |
| `coding_dev.jsonl` | 15 | `dp`, `das`, `das_count` |
| `coding_dp_only.jsonl` | 30 | `dp` only, to illustrate partial supervision |

Each line is a JSON object with a `note_id`, a `note_text` and the annotated
columns, which `eds.omop_dict2doc` copies to the matching `Doc._` extensions.
