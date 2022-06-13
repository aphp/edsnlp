# Measurements

The `eds.measurements` pipeline's role is to detect and normalise numerical measurements within a medical document.
We use simple regular expressions to extract and normalize measurements, and use `Measurement` classes to store them.

!!! warning

    The ``measurements`` pipeline is still in active development and has not been rigorously validated.
    If you come across a measurement expression that goes undetected, please file an issue !

## Scope

The `eds.measurements` pipeline can extract simple (eg `3cm`) measurements.
It can detect elliptic enumerations (eg `32, 33 et 34kg`) of measurements of the same type and split the measurements accordingly.

The normalized value can then be accessed via the `span._.value` attribute and converted on the fly to a desired unit.

The current pipeline annotates the following measurements out of the box:

| Measurement name | Example                |
| ---------------- | ---------------------- |
| `eds.size`       | `1m50`, `1.50m`        |
| `eds.weight`     | `12kg`, `1kg300`       |
| `eds.bmi`        | `BMI: 24`, `24 kg.m-2` |
| `eds.volume`     | `2 cac`, `8ml`         |

## Usage

```python
import spacy

nlp = spacy.blank("eds")
nlp.add_pipe(
    "eds.measurements", config=dict(measurements=["eds.size", "eds.weight", "eds.bmi"])
)

text = (
    "Le patient est admis hier, fait 1m78 pour 76kg. "
    "Les deux nodules bénins sont larges de 1,2 et 2.4mm. "
    "BMI: 24 "
)

doc = nlp(text)

measurements = doc.spans["measurements"]

measurements
# Out: [1m78, 76kg, 1,2, 2.4mm, 24]

measurements[0]
# Out: 1m78

str(measurements[0]._.value)
# Out: '1.78 m'

measurements[0]._.value.cm
# Out: 178.0

measurements[2]
# Out: 1,2

str(measurements[2]._.value)
# Out: '1.2 mm'

str(measurements[2]._.value.mm)
# Out: 1.2

measurements[4]
# Out: 24

str(measurements[4]._.value)
# Out: '24.0 kg_per_m2'

str(measurements[4]._.value.kg_per_m2)
# Out: 24.0
```

## Custom measurement

You can declare custom measurements by changing the patterns

```python
import spacy

nlp = spacy.blank("eds")
nlp.add_pipe(
    "eds.measurements",
    config=dict(
        measurements={
            # this name will be used to define the labels of the matched entities
            "my_custom_surface_measurement": {
                # This measurement unit is homogenous to square meters
                "unit": "m2",
                # To handle cases like "surface: 1.8" (implied m2), we can use
                # unitless patterns
                "unitless_patterns": [
                    {
                        "terms": ["surface", "aire"],
                        "ranges": [
                            {
                                "unit": "m2",
                                "min": 0,
                                "max": 9,
                            }
                        ],
                    }
                ],
            },
        }
    ),
)
```

## Declared extensions

The `eds.measurements` pipeline declares a single [spaCy extension](https://spacy.io/usage/processing-pipelines#custom-components-attributes) on the `Span` object,
the `value` attribute that is a `Measurement` instance.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter         | Explanation                                                                    | Default                                                              |
| ----------------- | --------------------------------------------------------------------------     | -------------------------------------------------------------------- |
| `measurements`    | A list or dict of the measurements to extract                                  | `["eds.size", "eds.weight", "eds.angle"]` |
| `units_config`    | A dict describing the units with lexical patterns, dimensions, scales, ...     | ... |
| `number_terms`    | A dict describing the textual forms of common numbers                          | ... |
| `stopwords`       | A list of stopwords that do not matter when placed between a unitless trigger  | ... |
| `unit_divisors`   | A list of terms used to divide two units (like: m / s)                         | ... |
| `ignore_excluded` | Whether to ignore excluded tokens for matching                                 | `False`                                                              |
| `attr`            | spaCy attribute to match on, eg `NORM` or `TEXT`                               | `"NORM"`                                                             |

## Authors and citation

The `eds.measurements` pipeline was developed by AP-HP's Data Science team.
