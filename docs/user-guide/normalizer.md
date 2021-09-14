# Normalizer

The `normalizer` pipeline's role is to apply normalization on the input text, in order to simplify the extraction of a terminology. The modification only impacts the `NORM` attribute, and therefore adheres to the non-destructive doctrine. In other words,

```python
nlp(text).text == text
```

remains true.

The normalizer can normalize the input text in three dimensions :

1. Move the text to lowercase.
2. Remove accents. We use a deterministic approach to avoid modifying the character-length of the text (ie `len(token.norm_) == len(token.text)`). Otherwise, regular expression matching would become a hassle.
3. Normalize apostrophes and quotation marks, which are often coded using special characters.

By default, all three normalizations are activated.

## Usage

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("normalizer")  # exposed via edsnlp.components

text = "Le patient est admis le 23 août 2021 pour une douleur à l'estomac."

doc = nlp(text)

doc[6]
# Out: août

doc[6].norm_
# Out: aout
```

## Authors and citation

This pipeline was developed at AP-HP's EDS by the Data Science team.
