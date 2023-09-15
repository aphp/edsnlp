# Scores Overview

EDS-NLP provides multiple matchers for typical scores (Charlson, SOFA...) found in clinical documents.
To extract a score, the matcher:

- extracts the score's name via the provided regular expressions
- extracts the score's _raw_ value via another set of RegEx
- normalize the score's value via a normalising function

## Available scores

| Component                | Description                |
|--------------------------|----------------------------|
| `eds.charlson`           | A Charlson score extractor |
| `eds.emergency_ccmu`     | A CCMU score extractor     |
| `eds.emergency_gemsa`    | A GEMSA score extractor    |
| `eds.emergency_priority` | A priority score extractor |
| `eds.sofa`               | A SOFA score extractor     |
| `eds.tnm`                | A TNM score extractor      |

## Implementing your own score

Using the `eds.score` pipeline, you only have to change its configuration in order to implement a _simple_ score extraction algorithm. As an example, let us see the configuration used for the `eds.charlson` pipe
The configuration consists of 4 items:

- `score_name`: The name of the score
- `regex`: A list of regular expression to detect the score's mention
- `value_extract`: A regular expression to extract the score's value in the context of the score's mention
- `score_normalization`: A function name used to normalise the score's _raw_ value

!!! note

    Functions passed as parameters to components need to be registered as follow

    ```python
    import spacy


    @spacy.registry.misc("score_normalization.charlson")
    def my_normalization_score(raw_score: str):
        # Implement some filtering here
        # Return None if you want the score to be discarded
        return normalized_score
    ```

The values used for the `eds.charlson` pipe are the following:

```python
import spacy


@spacy.registry.misc("score_normalization.charlson")
def score_normalization(extracted_score):
    """
    Charlson score normalization.
    If available, returns the integer value of the Charlson score.
    """
    score_range = list(range(0, 30))
    if (extracted_score is not None) and (int(extracted_score) in score_range):
        return int(extracted_score)


charlson_config = dict(
    score_name="charlson",
    regex=[r"charlson"],
    value_extract=r"charlson.*[\n\W]*(\d+)",
    score_normalization="score_normalization.charlson",
)
```
