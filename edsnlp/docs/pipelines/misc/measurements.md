# Measurements

The `eds.measurements` pipeline's role is to detect and normalise numerical measurements within a medical document.
We use simple regular expressions to extract and normalize measurements, and use `SimpleMeasurement` classes to store them.

## Scope

By default, the `eds.measurements` pipeline lets you match all measurements, i.e measurements in most units as well as unitless measurements. If a unit is not in our register,
then you can add It manually. If not, the measurement will be matched without Its unit.

If you prefer matching specific measurements only, you can create your own measurement config anda set `all_measurements` parameter to `False`. Nevertheless, some default measurements configs are already provided out of the box:

| Measurement name | Example                |
| ---------------- | ---------------------- |
| `eds.size`       | `1m50`, `1.50m`        |
| `eds.weight`     | `12kg`, `1kg300`       |
| `eds.bmi`        | `BMI: 24`, `24 kg.m-2` |
| `eds.volume`     | `2 cac`, `8ml`         |
| `eds.bool`       | `positive`, `negatif`  |

The normalized value can then be accessed via the `span._.value` attribute and converted on the fly to a desired unit (eg `span._.value.g_per_cl` or `span._.value.kg_per_m3` for a density).

The measurements that can be extracted can have one or many of the following characteristics:
- Unitless measurements
- Measurements with unit
- Measurements with range indication (escpecially < or >)
- Measurements with power

The measurement can be written in many complex forms. Among them, this pipe can detect:
- Measurements with range indication, numerical value, power and units in many different orders and separated by customizable stop words
- Composed units (eg `1m50`)
- Measurement with "unitless patterns", i.e some textual information next to a numerical value which allows us to retrieve a unit even if It is not written (eg in the text `Height: 80`, this pipe will a detect the numlerical value `80`and match It to the unit `kg`)
- Elliptic enumerations (eg `32, 33 et 34mol`) of measurements of the same type and split the measurements accordingly

## Usage

This pipe works better with `eds.dates` and `eds.tables` pipe at the same time. These pipes let `eds.measurements` skip dates as measurements and make a specific matching for each table, benefitting of the structured data.

The matched measurements are labeled with a default measurement name if available (eg `eds.size`), else `eds.measurement` if any measure is linked to the dimension of the measure's unit and if `all_measurements` is set to `True`.

As said before, each matched measurement can be accessed via the `span._.value`. This gives you a `SimpleMeasurement` object with the following attributes :
- `value_range` ("<", "=" or ">")
- `value`
- `unit`
- `registry` (This attribute stores the entire unit config like the link between each unit, Its dimension like `length`, `quantity of matter`...)

`SimpleMeasurement` objects are especially usefull when converting measurements to an other specified unit with the same dimension (eg densities stay densities). To do so, simply call your `SimpleMeasurement` followed by `.` + name of the usual unit abbreviation with `per` and `_` as separators (eg `object.kg_per_dm3`, `mol_per_l`, `g_per_cm2`).

Moreover, for now, `SimpleMeasurement` objects can be manipulated with the following operations:
- compared with an other `SimpleMeasurement` object with the same dimension with automatic conversion (eg a density in kg_per_m3 and a density in g_per_l)
- summed with an other `SimpleMeasurement` object with the same dimension with automatic conversion
- substracted with an other `SimpleMeasurement` object with the same dimension with automatic conversion

Note that for all operations listed above, different `value_range` attributes between two units do not matter: by default, the `value_range` of the first measurement is kept.

Below is a complete example on a use case where we want to extract size, weigth and bmi measurements a simple text.

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

You can declare custom measurements by changing the patterns.

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
the `value` attribute that is a `SimpleMeasurement` instance.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter                | Explanation                                                                      | Default                                                                   |
| ------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `measurements`           | A list or dict of the measurements to extract                                    | `None` # Extract measurements from all units                              |
| `units_config`           | A dict describing the units with lexical patterns, dimensions, scales, ...       | ... # Config of mostly all commonly used units                            |
| `number_terms`           | A dict describing the textual forms of common numbers                            | ... # Config of mostly all commonly used textual forms of common numbers  |
| `value_range_terms`      | A dict describing the textual forms of ranges ("<", "=" or ">")                  | ... # Config of mostly all commonly used range terms                      |
| `stopwords_unitless`     | A list of stopwords that do not matter when placed between a unitless trigger    | `["par", "sur", "de", "a", ":", ",", "et"]`                               |
| `stopwords_measure_unit` | A list of stopwords that do not matter when placed between a measure and a unit  | `["|", "¦", "…", "."]`                                                    |
| `measure_before_unit`    | A bool to tell if the numerical value is usually placed before the unit          | `["par", "sur", "de", "a", ":", ",", "et"]`                               |
| `unit_divisors`          | A list of terms used to divide two units (like: m / s)                           | `["/", "par"]`                                                            |
| `ignore_excluded`        | Whether to ignore excluded tokens for matching                                   | `False`                                                                   |
| `attr`                   | spaCy attribute to match on, eg `NORM` or `TEXT`                                 | `"NORM"`                                                                  |

## Authors and citation

The `eds.measurements` pipeline was developed by AP-HP's Data Science team.
