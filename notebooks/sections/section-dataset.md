---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%reload_ext autoreload
%autoreload 2
```

```python
import pandas as pd
```

```python
import os
```

```python
import context
```

```python
from nlptools.utils.brat import BratConnector
```

# Sections dataset


Réutilisation du [travail réalisé par Ivan Lerner à l'EDS](https://gitlab.eds.aphp.fr/IvanL/section_dataset).

```python
data_dir = '../../data/section_dataset/'
```

```python
brat = BratConnector(data_dir)
```

```python
texts, annotations = brat.get_brat()
```

```python
df = annotations[['lexical_variant']].drop_duplicates()
```

```python
df['section'] = ''
```

```python
df.to_csv('sections.tsv', sep='\t', index=False)
```

```python
annotated = pd.read_csv('sections.tsv', sep='\t')
```

```python
annotated.to_csv('annotated_sections.csv', index=False)
```

```python
annotated = pd.read_excel('sections.xlsx', sheet_name='Annotation', engine='openpyxl')
```

```python
annotated.columns = ['lexical_variant', 'section', 'keep', 'comment']
```

```python
annotated.keep = annotated.keep.fillna('Oui') == 'Oui'
```

```python
annotated = annotated.query('keep')[['lexical_variant', 'section']]
```

```python
annotated.merge(annotations, on='lexical_variant').section.value_counts()
```

```python
annotated.lexical_variant = annotated.lexical_variant.str.lower()
```

```python
annotated_unnaccented = annotated.copy()
```

```python
from unidecode import unidecode
```

```python
annotated_unnaccented.lexical_variant = annotated_unnaccented.lexical_variant.apply(unidecode)
```

```python
# annotated = pd.concat([annotated, annotated_unnaccented])
annotated = annotated_unnaccented
```

```python
annotated = annotated.drop_duplicates()
```

```python
annotated = annotated.sort_values(['lexical_variant', 'section'])
```

```python
annotated
```

```python
annotated = annotated.drop_duplicates()
```

```python
sections = {
    section.replace(' ', '_'): list(annotated.query('section == @section').lexical_variant)
    for section in annotated.section.unique()
}
```

```python
for k, v in sections.items():
    print(unidecode(k.replace(' ', '_')), '=', v)
    print()
```

```python
sections = {
    section: unidecode(section.replace(' ', '_'))
    for section in annotated.section.unique()
}
```

```python
for k, v in sections.items():
    print(f"{repr(k)}: {v},")
```

```python

```
