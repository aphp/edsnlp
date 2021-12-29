# Normalisation

The normalisation scheme used by EDS-NLP adheres to the non-destructive doctrine. In other words,

```python
nlp(text).text == text
```

is always true.

To achieve this, the input text is never modified. Instead, in EDS-NLP the normalisation is done in two axes:

1. The textual representation is modified using the `NORM` attribute **only** ;
2. Pipelines can mark tokens as _excluded_ by setting the extension `Token._.excluded` to `True`. It enables downstream matchers to skip excluded tokens.

The normalizer can act on the input text in four dimensions :

1. Move the text to [lowercase](#lowercase).
2. Remove [accents](#accents). We use a deterministic approach to avoid modifying the character-length of the text, which helps for RegEx matching.
3. Normalize [apostrophes and quotation marks](#apostrophes-and-quotation-marks), which are often coded using special characters.
4. Remove [pollutions](#pollution).

By default, the first four normalizations are activated. The `endlines` normalisation requires training a model, refer to [the dedicated page for more information](endlines.md).

## Utilities

To simplify the use of the normalisation output, we provide the `get_text` utility function. It computes the textual representation for a `Span` or `Doc` object.

Moreover, every span exposes a `normalized_variant` extension getter, which computes the normalised representation of an entity on the fly.

## Pipelines

### Lowercase

The `lowercase` pipeline transforms every token to lowercase. It is not configurable.

Consider the following example :

```python
import spacy
from edsnlp.matchers.utils import get_text
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

get_text(doc, attr="NORM")
# Out: pneumopathie à nbnbwbwbnbwbnbnbnbwbw `coronavirus'
```

### Accents

The `accents` pipeline removes accents. To avoid edge cases, the uses a specified list of accentuated characters and there unaccentuated representation, making it more predictable than using a library such as `unidecode`.

Consider the following example :

```python
import spacy
from edsnlp.matchers.utils import get_text
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

get_text(doc, attr="NORM")
# Out: Pneumopathie a NBNbWbWbNbWbNBNbNbWbW `coronavirus'
```

### Apostrophes and quotation marks

Apostrophes and quotation marks can be encoded using unpredictable special characters. The `quotes` component transforms every such special character to `'` and `"`, respectively.

Consider the following example :

```python
import spacy
from edsnlp.matchers.utils import get_text
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

get_text(doc, attr="NORM")
# Out: Pneumopathie à NBNbWbWbNbWbNBNbNbWbW 'coronavirus'
```

### Pollution

The pollution pipeline uses a set of regular expressions to detect pollutions (irrelevant non-medical text that hinders text processing). Corresponding tokens are marked as excluded (by setting `Token._.excluded` to `True`), enabling the use of the phrase matcher.

Consider the following example :

```python
import spacy
from edsnlp.matchers.utils import get_text
from edsnlp import components

config = dict(
    lowercase=False,
    accents=True,
    quotes=False,
    pollution=True,
    endlines=False,
)

nlp = spacy.blank("fr")
nlp.add_pipe("normalizer", config=config)

text = "Pneumopathie à NBNbWbWbNbWbNBNbNbWbW `coronavirus'"

doc = nlp(text)

get_text(doc, attr="NORM")
# Out: Pneumopathie a NBNbWbWbNbWbNBNbNbWbW `coronavirus'

get_text(doc, attr="TEXT", ignore_excluded=True)
# Out: Pneumopathie à `coronavirus'
```

This example above shows that the normalisation scheme works on two axes: non-destructive text modification and exclusion of tokens.
The two are independent: a matcher can use the `NORM` attribute but keep excluded tokens, and conversely, match on `TEXT` while ignoring excluded tokens.

### New lines

The `endlines` pipeline classifies newline characters as actual end of lines or mere spaces. In the latter case, the token is removed from the normalised document.

See the [dedicated documentation](endlines.md) for more detail.

## Usage

As seen in the previous examples, the normalisation is handled by the single `normalizer` pipeline. The following code snippet is complete, and should run as is.

```python
import spacy
from edsnlp.matchers.utils import get_text
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("normalizer")

# Notice the special character used for the apostrophe and the quotes
text = "Le patient est admis à l'hôpital le 23 août 2021 pour une douleur ʺaffreuse” à l`estomac."

doc = nlp(text)

get_text(doc, attr="NORM")
# Out: le patient est admis a l'hopital le 23 aout 2021 pour une douleur "affreuse" a l'estomac
```

## Authors and citation

The `normalizer` pipeline was developed at the Data and Innovation unit, IT department, AP-HP.
