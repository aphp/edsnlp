---
jupyter:
  jupytext:
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: 'Python 3.9.5 64-bit (''.env'': venv)'
    name: python3
---

```python
import context
```

```python
from edsnlp.pipelines.misc.dates import Dates, terms
```

```python
from datetime import datetime
```

```python
import spacy
```

# Date detection

```python
text = (
    "Le patient est arrivé le 23 août (23/08/2021). "
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
dates = Dates(
    nlp,
    absolute=terms.absolute,
    relative=terms.relative,
    no_year=terms.no_year,
)
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
    print(f"{span.text:<20}  {span._.date}")
```

Lorsque la date du document n'est pas connue, le label des dates relatives (hier, il y a quinze jours, etc) devient `TD±<nb-de-jours>`

Si on renseigne l'extension `note_datetime` :

```python
doc._.note_datetime = datetime(2020, 10, 10)
```

```python
dates(doc)
```

```python
print(f"{'expression':<20}  label")
print(f"{'----------':<20}  -----")

for span in doc.spans['dates']:
    print(f"{span.text:<20}  {span._.date}")
```
