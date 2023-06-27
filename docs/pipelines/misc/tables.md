# Tables

The `eds.tables` pipeline's role is to detect tables present in a medical document.
We use simple regular expressions to extract tables like text.

## Usage

This pipe lets you match different forms of tables. They can have a frame or not, rows can be spread on multiple consecutive lines (in case of a bad parsing for example)... You can also indicate the presence of headers with the `col_names` and `row_names` boolean parameters.

Each matched table is returned as a `Span` object. You can then access to an equivalent dictionnary formatted table with `table` extension or use `to_pandas_table()` to get the equivalent pandas DataFrame. The key of the dictionnary is determined as folowed:
- If `col_names` is True, then, the dictionnary keys are the names of the columns (str).
- Elif `row_names` is True, then, the dictionnary keys are the names (str).
- Else the dictionnary keys are indexes of the columns (int).

`to_pandas_table()` can be customised with `as_spans` parameter. If set to `True`, then the pandas dataframe will contain the cells as spans, else the pandas dataframe will contain the cells as raw strings.

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
df = table._.to_pd_table(as_spans=False)
type(df)
# >> pandas.core.frame.DataFrame
```
The pd DataFrame:
|      | 0           | 1        | 2      | 3         |
| ---: | :---------- | :------- | :----- | :-------- |
|    0 | Leucocytes  | x10*9/L  | 4.97   | 4.09-11   |
|    1 | Hématies    | x10*12/L | 4.68   | 4.53-5.79 |
|    2 | Hémoglobine | g/dL     | 14.8   | 13.4-16.7 |
|    3 | Hématocrite | %        | 44.2   | 39.2-48.6 |
|    4 | VGM         | fL       | 94.4 + | 79.6-94   |
|    5 | TCMH        | pg       | 31.6   | 27.3-32.8 |
|    6 | CCMH        | g/dL     | 33.5   | 32.4-36.3 |
|    7 | Plaquettes  | x10*9/L  | 191    | 172-398   |
|    8 | VMP         | fL       | 11.5 + | 7.4-10.8  |

## Declared extensions

The `eds.tables` pipeline declares two [spaCy extension](https://spacy.io/usage/processing-pipelines#custom-components-attributes) on the `Span` object. The first one is `to_pd_table()` method which returns a parsed pandas version of the table. The second one is `table` which contains the table stored as a dictionnary containing cells as `Span` objects.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter         | Explanation                                      | Default                |
| ----------------- | ------------------------------------------------ | ---------------------- |
| `tables_pattern`  | Pattern to identify table spans                  | `rf"(\b.*{sep}.*\n)+"` |
| `sep_pattern`     | Pattern to identify column separation            | `r"¦"`                 |
| `ignore_excluded` | Ignore excluded tokens                           | `True`                 |
| `attr`            | spaCy attribute to match on, eg `NORM` or `TEXT` | `"TEXT"`               |

## Authors and citation

The `eds.tables` pipeline was developed by AP-HP's Data Science team.
