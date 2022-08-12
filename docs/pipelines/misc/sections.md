# Sections

Detected sections are :

- `allergies`
- `antécédents`
- `antécédents familiaux`
- `traitements entrée`
- `conclusion`
- `conclusion entrée`
- `habitus`
- `correspondants`
- `diagnostic`
- `données biométriques entrée`
- `examens`
- `examens complémentaires`
- `facteurs de risques`
- `histoire de la maladie`
- `actes`
- `motif`
- `prescriptions`
- `traitements sortie`
- `evolution`
- `modalites sortie`
- `vaccinations`
- `introduction`


<!--
  | Section                       | Description |
  | ----------------------------- | ----------- |
  | `allergies`                   |             |
  | `antécédents`                 |             |
  | `antécédents familiaux`       |             |
  | `traitements entrée`          |             |
  | `conclusion`                  |             |
  | `conclusion entrée`           |             |
  | `habitus`                     |             |
  | `correspondants`              |             |
  | `diagnostic`                  |             |
  | `données biométriques entrée` |             |
  | `examens`                     |             |
  | `examens complémentaires`     |             |
  | `facteurs de risques`         |             |
  | `histoire de la maladie`      |             |
  | `actes`                       |             |
  | `motif`                       |             |
  | `prescriptions`               |             |
  | `traitements sortie`          |             |
  | `evolution`                   |             |
  | `modalites sortie`            |             |
  | `vaccinations`                |             |
  | `introduction`                |             | -->

<!-- ![Section extraction](/resources/sections.svg){ align=right width="35%"} -->

The pipeline extracts section title. A "section" is then defined as the span of text between two titles.

Remarks :
- section `introduction` corresponds to the span of text between the header "COMPTE RENDU D'HOSPITALISATION" (usually denoting the beginning of the document) and the title of the following detected section
- this pipeline works well for hospitalization summaries (CRH), but not necessarily for all types of documents (in particular for emergency or scan summaries CR-IMAGERIE)

!!! warning "Use at your own risks"

    Should you rely on `eds.sections` for critical downstream tasks, make sure to validate the pipeline to make sure that the component works.
    For instance, the `eds.history` pipeline can use sections to make its predictions, but that possibility is deactivated by default.

## Usage

The following snippet detects section titles. It is complete and can be run _as is_.

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.sections")

text = "CRU du 10/09/2021\n" "Motif :\n" "Patient admis pour suspicion de COVID"

doc = nlp(text)

doc.spans["section_titles"]
# Out: [Motif]
```

## Configuration

The pipeline can be configured using the following parameters :

| Parameter         | Explanation                                      | Default                           |
| ----------------- | ------------------------------------------------ | --------------------------------- |
| `sections`        | Sections patterns                                | `None` (use pre-defined patterns) |
| `add_patterns`    | Whether add endlines patterns                    | `False`                           |
| `attr`            | spaCy attribute to match on, eg `NORM` or `TEXT` | `"NORM"`                          |
| `ignore_excluded` | Whether to ignore excluded tokens                | `True`                            |

## Declared extensions

The `eds.sections` pipeline adds two fields to the `doc.spans` attribute :

1. The `section_titles` key contains the list of all section titles extracted using the list declared in the `terms.py` module.
2. The `sections` key contains a list of sections, ie spans of text between two section titles (or the last title and the end of the document).

## Authors and citation

The `eds.sections` pipeline was developed by AP-HP's Data Science team.
