---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Python 3.9.5 64-bit ('.env')
    name: python3
---

```python
import context
```

```python
from edsnlp.pipelines.dates import Dates, terms
```

```python
import spacy
```

# Date detection

```python
text = (
    "Le patient est arrivé le 23 août (23/08). "
    "Il dit avoir eu mal au ventre hier. "
    "L'année dernière, on lui avait prescrit du doliprane."
)
```

```python
nlp = spacy.blank('fr')
```

```python
doc = nlp(text)
```

```python
dates = Dates(nlp, absolute=terms.absolute, relative=terms.relative)
```

```python
dates(doc)
```

```python
doc.spans
```

```python
print(f"{'expression':<20}  label")
print(f"{'----------':<20}  -----")

for span in doc.spans['dates']:
    print(f"{span.text:<20}  {span.label_}")
```

Lorsque la date du document n'est pas connue, le label des dates relatives (hier, il y a quinze jours, etc) devient `TD±<nb-de-jours>`
