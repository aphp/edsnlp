# The different problems found

## 1. Installation of requirements :

when doing `pip install -r requirements-{...}.txt`, two are not compatible  : if `dev`then `doc`, the following red ERROR appear :

<span style="color:red">
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
flake8 4.0.1 requires importlib-metadata<4.3; python_version < "3.8", but you have importlib-metadata 4.11.3 which is incompatible.
</span>


## 2. Get_text in the doc

in `https://aphp.github.io/edsnlp/latest/pipelines/core/normalisation/`, the codes given raise an error because `get_text` needs the attribute `ignore_excluded`. 

Exemple of wrong code :

```python
import spacy
from edsnlp.matchers.utils import get_text

config = dict(
    lowercase=False,
    accents=True,   
    quotes=False,
    pollution=True,
)

nlp = spacy.blank("fr")
nlp.add_pipe("eds.normalizer", config=config)

text = "Pneumopathie à NBNbWbWbNbWbNBNbNbWbW `coronavirus'"

doc = nlp(text)

get_text(doc, attr="NORM")
# Out: Pneumopathie a NBNbWbWbNbWbNBNbNbWbW `coronavirus'

get_text(doc, attr="TEXT", ignore_excluded=True)
# Out: Pneumopathie à `coronavirus'
```


(Only the lase `get_text` works)



In `edsnlp.matchers.utils.text.py`, the doc still says that it's False by default :

```python
def get_text(
    doclike: Union[Doc, Span],
    attr: str,
    ignore_excluded: bool,
) -> str:
    """
    Get text using a custom attribute, possibly ignoring excluded tokens.

    Parameters
    ----------
    doclike : Union[Doc, Span]
        Doc or Span to get text from.
    attr : str
        Attribute to use.
    ignore_excluded : bool
        Whether to skip excluded tokens, by default False

    Returns
    -------
    str
        Extracted text.
    """
```

## 3. Endlines
in `https://aphp.github.io/edsnlp/latest/pipelines/core/endlines/` :

This piece of code doesn't work :
```python
import spacy
from edsnlp.pipelines.endlines.endlinesmodel import EndLinesModel
import pandas as pd
from spacy import displacy`
```

because `endlines` is in `edsnlp.pipelines.core`, so this piece of code works instead :

```python
import spacy
from edsnlp.pipelines.core.endlines.endlinesmodel import EndLinesModel
import pandas as pd
from spacy import displacy
``` 


## 4. Detecting Reason of Hospitalisation

in `https://aphp.github.io/edsnlp/latest/tutorials/reason/`

Globally the outputs are wrong


In the doc :
```python
# ↑ Omitted code above ↑

reason = doc.spans["reasons"][0]
reason
# Out: hospitalisé du 11/08/2019 au 17/08/2019 pour attaque d'asthme.
```

On my computer :

```python
# ↑ Omitted code above ↑

reason = doc.spans["reasons"][0]
reason
# Out: MOTIF D'HOSPITALISATION
# Out: Monsieur Dupont Jean Michel, de sexe masculin, âgée de 39 ans,
# Out: née le 23/11/1978, a été hospitalisé du 11/08/2019 au 17/08/2019
# Out: pour attaque d'asthme.
# Out: 
# Out: ANTÉCÉDENTS
```



In the doc 

```python
for e in doc.ents:
    print(e.start, e, e._.is_reason)

# Out: 42 asthme True
# Out: 54 asthme False
```

On my computer :

```python
for e in doc.ents:
    print(e.start, e, e._.is_reason)

# Out: 43 asthme True
# Out: 55 asthme False
```

# Little proposition for Hypothesis and Family

Instead of having a boolean, returning a score of certainty (for example : 0 = maybe, 0.5 = quite certaint, 1 = certaint) or the word that implied hypothesis (like for the reason detecion).

For Family the same but returning the member of the family.