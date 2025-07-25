# Miscellaneous

This section regroups components that extract information that can be used by other components, but have little medical value in itself.

For instance, the date detection and normalisation pipeline falls in this category.

## Available components

<!-- --8<-- [start:components] -->

| Component                | Description                                             |
|--------------------------|---------------------------------------------------------|
| `eds.dates`              | Date extraction and normalisation                       |
| `eds.consultation_dates` | Identify consultation dates                             |
| `eds.quantities`         | Quantity extraction and normalisation                   |
| `eds.sections`           | Section detection                                       |
| `eds.reason`             | Rule-based hospitalisation reason detection             |
| `eds.tables`             | Tables detection                                        |
| `eds.split`              | Doc splitting                                           |
| `eds.explode`            | Explode entities between multiples copies of a document |

<!-- --8<-- [end:components] -->
