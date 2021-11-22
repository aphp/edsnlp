# Normalizer

The `normalizer` pipeline's role is to apply normalization on the input text, in order to simplify the extraction of a terminology. The modification only impacts a custom attribute, and therefore adheres to the non-destructive doctrine. In other words,

```python
nlp(text).text == text
```

remains true.

The normalizer can normalize the input text in five dimensions :

1. Move the text to lowercase.
2. Remove accents. We use a deterministic approach to avoid modifying the character-length of the text.
3. Normalize apostrophes and quotation marks, which are often coded using special characters.
4. Remove pollutions.
5. Transform newline characters to spaces.

By default, the first four normalizations are activated. The `endlines` normalisation requires training a model, refer to [the dedicated page for more information](endlines.md).

To enable both regular expressions and phrase matching, the `normalizer` pipeline generates a new Spacy `Doc` object, which populates the `Doc._.normalized` extension. This strategy lets us arbitrarily modify tokens, and even remove tokens altogether. We provide helper methods to efficiently go back and forth between the original document and its normalised version :

```python
# Create a span in the normalized document
normalized_span = Span(doc._.normalized, 4, 8, label="norm")

# Go back to the original document
start = doc._.norm2original[normalized_span.start]
end = doc._.norm2original[normalized_span.end]

original_span = Span(doc, start, end, label=normalized_span.label)

original_span.label_
# Out: 'norm'
```

## Pipelines

### Lowercase

The `lowercase` pipeline transforms every token to lowercase. It is not configurable.

Consider the following example :

```python
import spacy
from edsnlp import components

config = dict(
    lowercase=True,
    accents=False,
    quotes=False,
    pollution=False,
    endlines=False,
)

nlp = spacy.blank("fr")
nlp.add_pipe("normalizer", config=config)

text = "Pneumopathie à NBNbWbWbNbWbNBNbNbWbW `coronavirus'"

doc = nlp(text)

doc._.normalized
# Out: pneumopathie à nbnbwbwbnbwbnbnbnbwbw `coronavirus'
```

### Accents

The `accents` pipeline removes accents. To avoid edge cases, the uses a specified list of accentuated characters and there unaccentuated representation, making it more predictable than using a library such as `unidecode`.

Consider the following example :

```python
import spacy
from edsnlp import components

config = dict(
    lowercase=False,
    accents=True,
    quotes=False,
    pollution=False,
    endlines=False,
)

nlp = spacy.blank("fr")
nlp.add_pipe("normalizer", config=config)

text = "Pneumopathie à NBNbWbWbNbWbNBNbNbWbW `coronavirus'"

doc = nlp(text)

doc._.normalized
# Out: Pneumopathie a NBNbWbWbNbWbNBNbNbWbW `coronavirus'
```

### Apostrophes and quotation marks

Apostrophes and quotation marks can be encoded using unpredictable special characters. The `quotes` component transforms every such special character to `'` and `"`, respectively.

Consider the following example :

```python
import spacy
from edsnlp import components

config = dict(
    lowercase=False,
    accents=False,
    quotes=True,
    pollution=False,
    endlines=False,
)

nlp = spacy.blank("fr")
nlp.add_pipe("normalizer", config=config)

text = "Pneumopathie à NBNbWbWbNbWbNBNbNbWbW `coronavirus'"

doc = nlp(text)

doc._.normalized
# Out: Pneumopathie à NBNbWbWbNbWbNBNbNbWbW 'coronavirus'
```

### Pollution

The pollution pipeline uses a set of regular expressions to detect pollutions (irrelevant non-medical text that hinders text processing). Corresponding tokens are simply removed from the normalized version of the document, enabling the use of the phrase matcher.

Consider the following example :

```python
import spacy
from edsnlp import components

config = dict(
    lowercase=False,
    accents=False,
    quotes=False,
    pollution=True,
    endlines=False,
)

nlp = spacy.blank("fr")
nlp.add_pipe("normalizer", config=config)

text = "Pneumopathie à NBNbWbWbNbWbNBNbNbWbW `coronavirus'"

doc = nlp(text)

doc._.normalized
# Out: Pneumopathie à `coronavirus'
```

### New lines

The `endlines` pipeline classifies newline characters as actual end of lines or mere spaces. In the latter case, the token is removed from the normalised document.

See the [dedicated documentation](endlines.md) for more detail.

## Usage

As seen in the previous examples, the normalisation is handled by the single `normalizer` pipeline. The following code snippet is complete, and should run as is.

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("normalizer")

# Notice the special character used for the apostrophe and the quotes
text = "Le patient est admis à le 23 août 2021 pour une douleur ʺaffreuse” à l`estomac."

doc = nlp(text)

doc._.normalized
# Out: le patient est admis a l'hopital le 23 aout 2021 pour une douleur "affreuse" a l'estomac
```

Here, `doc._.normalized` is actually a new Spacy `Doc` object.

## Authors and citation

The `normalizer` pipeline was developed at the Data and Innovation unit, IT department, AP-HP.
