# Tables

The `eds.tables` pipeline's role is to detect tables present in a medical document.
We use simple regular expressions to extract tables like text.

## Usage

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.tables")

text = """
SERVICE
MEDECINE INTENSIVE –
REANIMATION
Réanimation / Surveillance Continue
Médicale

COMPTE RENDU D'HOSPITALISATION du 05/06/2020 au 10/06/2020
Madame DUPONT Marie, née le 16/05/1900, âgée de 20 ans, a été hospitalisée en réanimation du
05/06/1920 au 10/06/1920 pour intoxication médicamenteuse volontaire.


Examens complémentaires
Hématologie
Numération
Leucocytes ¦x10*9/L ¦4.97 ¦4.09-11
Hématies ¦x10*12/L¦4.68 ¦4.53-5.79
Hémoglobine ¦g/dL ¦14.8 ¦13.4-16.7
Hématocrite ¦% ¦44.2 ¦39.2-48.6
VGM ¦fL ¦94.4 + ¦79.6-94
TCMH ¦pg ¦31.6 ¦27.3-32.8
CCMH ¦g/dL ¦33.5 ¦32.4-36.3
Plaquettes ¦x10*9/L ¦191 ¦172-398
VMP ¦fL ¦11.5 + ¦7.4-10.8

Sur le plan neurologique : Devant la persistance d'une confusion à distance de l'intoxication au
...

2/2Pat : <NOM> <Prenom>|F |<date> | <ipp> |Intitulé RCP

"""

doc = nlp(text)

# A table span
table = doc.spans["tables"][0]
# Leucocytes ¦x10*9/L ¦4.97 ¦4.09-11
# Hématies ¦x10*12/L¦4.68 ¦4.53-5.79
# Hémoglobine ¦g/dL ¦14.8 ¦13.4-16.7
# Hématocrite ¦% ¦44.2 ¦39.2-48.6
# VGM ¦fL ¦94.4 + ¦79.6-94
# TCMH ¦pg ¦31.6 ¦27.3-32.8
# CCMH ¦g/dL ¦33.5 ¦32.4-36.3
# Plaquettes ¦x10*9/L ¦191 ¦172-398
# VMP ¦fL ¦11.5 + ¦7.4-10.8

# Convert span to Pandas table
df = table._.to_pd_table
type(df)
# >> pandas.core.frame.DataFrame
```

## Declared extensions

The `eds.tables` pipeline declares one [spaCy extension](https://spacy.io/usage/processing-pipelines#custom-components-attributes) on the `Span` object: the `to_pd_table` attribute contains a parsed pandas version of the table.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter        | Explanation                                      | Default                           |
|------------------|--------------------------------------------------|-----------------------------------|
| `tables_pattern`       | Pattern to identify table spans     | `rf"(\b.*{sep}.*\n)+"` |
| `sep_pattern`       |Pattern to identify column separation              | `r"¦"` |
| `ignore_excluded`      | Ignore excluded tokens     | `True`  |
| `attr`           | spaCy attribute to match on, eg `NORM` or `TEXT` | `"TEXT"`                          |

## Authors and citation

The `eds.tables` pipeline was developed by AP-HP's Data Science team.
