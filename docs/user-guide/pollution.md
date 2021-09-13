# Removing pollution

Many documents from the EDS present uninformative snippets of text that hamper their readability.

An example of such "pollution" found in EDS medical note (here the transcription of a bar code):

```
NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbN...
```

or

```
Information aux patients : vous êtes informé(e) que les données à caractère personnel vous concernant, recueillies dans le cadre de la gestion administrative et de votre
prise en charge à l’hôpital, peuvent être utilisées à des fins de recherche dans le domaine de la santé, sous la responsabilité de l’AP-HP. Notamment, un Entrepôt de
Données de Santé (EDS) a été créé afin de permettre la réalisation de recherches non interventionnelles sur données, d’études de faisabilité des essais cliniques et
d’études de pilotage de l’activité hospitalière. Pour plus d’informations relatives à chaque recherche, aux données utilisées, aux destinataires des données, aux durées de
conservation des données et aux modalités d’exercice de vos droits, vous pouvez consulter le portail d’information de l’EDS à l’adresse http://recherche.aphp.fr/eds. Pour
vous opposer à l’utilisation des données vous concernant à des fins de recherche, vous pouvez vous adresser au bureau des usagers ou directeur de l’hôpital où vous
avez été pris en charge ou remplir le formulaire d’opposition électronique disponible à l’adresse http://recherche.aphp.fr/eds/droit-opposition.
```

The latter example is not a pollution _per se_, but the snippet is present in some form or another in more than 20% of AP-HP documents edited after August 2017, and bears no relevant information. To wit, the project `Embeddings` decided to remove this paragraph from their training set, to avoid skewing the language model towards it.

## Adding pollution patterns

By default, the `pollution` pipeline looks for regular expressions representing a few known pollution sources (see [source code for details](https://gitlab.eds.aphp.fr/equipedatascience/nlptools/-/blob/master/nlptools/rules/pollution/terms.py)).

## Non-destruction

All text normalisation in EDS-NLP is non-destructive, ie

```python
nlp(text).text == text
```

is always true.

Hence, the strategy chosen for the pollution pipeline is the following:

1. Tag, **but do not remove**, pollutions on the `Token._.pollution` extension.
2. Propose a `Doc._.clean_` extension, to retrieve the cleaned text.

## Recipes

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("pollution")  # exposed via edsnlp.components

text = (
    "Le patient est admis pour des douleurs dans le bras droit, mais n'a pas de problème de locomotion. "
    "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNB"
    "WbNbWbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "Douleurs dans le bras droit."
)

doc = nlp(text)
```

## Working on the cleaned text

Should you need to implement a pipeline using the cleaned version of the documents, the Pollution pipeline also exposes a `Doc._.clean_char_span` method to realign annotations made on the clean text with the original document.

```python
clean = nlp(doc._.clean)
span = clean[27:28]

doc._.clean_[span.start_char : span.end_char]
# Out: 'rhume'

doc.text[span.start_char : span.end_char]
# Out: 'bWbNb'

doc._.char_clean_span(span.start_char, span.end_char)
# Out: rhume
```
