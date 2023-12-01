# ADICAP

The `eds.adicap` pipeline component matches the ADICAP codes. It was developped to work on anapathology reports.

!!! warning "Document type"

    It was developped to work on anapathology reports.

    We recommend also to use the `eds` language (`spacy.blank("eds")`)

The compulsory characters of the ADICAP code are identified and decoded.
These characters represent the following attributes:



| Field [en]            | Field [fr]                       | Attribute             |
|-----------------------|----------------------------------|-----------------------|
| Sampling mode         | Mode de prelevement              | sampling_mode         |
| Technic               | Type de technique                | technic               |
| Organ and regions     | Appareils, organes et régions    | organ                 |
| Pathology             | Pathologie générale              | pathology             |
| Pathology type        | Type de la pathologie            | pathology_type        |
| Behaviour type        | Type de comportement             | behaviour_type        |


The pathology field takes 4 different values corresponding to the 4 possible interpretations of the ADICAP code, which are : "PATHOLOGIE GÉNÉRALE NON TUMORALE", "PATHOLOGIE TUMORALE", "PATHOLOGIE PARTICULIERE DES ORGANES" and "CYTOPATHOLOGIE".

Depending on the pathology value the behaviour type meaning changes, when the pathology is tumoral then it describes the malignancy of the tumor.

For further details about the ADICAP code follow this [link](https://smt.esante.gouv.fr/wp-json/ans/terminologies/document?terminologyId=terminologie-adicap&fileName=cgts_sem_adicap_fiche-detaillee.pdf).

## Usage

<!-- no-check -->

```python
import spacy

nlp = spacy.blank("eds")
nlp.add_pipe("eds.sentences")
nlp.add_pipe("eds.adicap")

text = """"
COMPTE RENDU D’EXAMEN

Antériorité(s) :  NEANT


Renseignements cliniques :
Contexte d'exploration d'un carcinome canalaire infiltrant du quadrant supéro-externe du sein droit. La
lésion biopsiée ce jour est située à 5,5 cm de la lésion du quadrant supéro-externe, à l'union des
quadrants inférieurs.


Macrobiopsie 10G sur une zone de prise de contraste focale à l'union des quadrants inférieurs du
sein droit, mesurant 4 mm, classée ACR4

14 fragments ont été communiqués fixés en formol (lame n° 1a et lame n° 1b) . Il n'y a pas eu
d'échantillon congelé. Ces fragments ont été inclus en paraffine en totalité et coupés sur plusieurs
niveaux.
Histologiquement, il s'agit d'un parenchyme mammaire fibroadipeux parfois légèrement dystrophique
avec quelques petits kystes. Il n'y a pas d'hyperplasie épithéliale, pas d'atypie, pas de prolifération
tumorale. On note quelques suffusions hémorragiques focales.

Conclusion :
Légers remaniements dystrophiques à l'union des quadrants inférieurs du sein droit.
Absence d'atypies ou de prolifération tumorale.

Codification :   BHGS0040
"""

doc = nlp(text)

doc.ents
# Out: (BHGS0040,)

ent = doc.ents[0]

ent.label_
# Out: adicap

ent._.adicap.dict()
# Out: {'code': 'BHGS0040',
# 'sampling_mode': 'BIOPSIE CHIRURGICALE',
# 'technic': 'HISTOLOGIE ET CYTOLOGIE PAR INCLUSION',
# 'organ': "SEIN (ÉGALEMENT UTILISÉ CHEZ L'HOMME)",
# 'pathology': 'PATHOLOGIE GÉNÉRALE NON TUMORALE',
# 'pathology_type': 'ETAT SUBNORMAL - LESION MINEURE',
# 'behaviour_type': 'CARACTERES GENERAUX'}
```

## Configuration

The pipeline can be configured using the following parameters :

::: edsnlp.pipelines.ner.adicap.factory.create_component
    options:
        only_parameters: true


## Authors and citation

The `eds.adicap` pipeline was developed by AP-HP's Data Science team.
The codes were downloaded from the website of 'Agence du numérique en santé' [@terminologie-adicap] ("Thésaurus de la codification ADICAP - Index raisonné des lésions")

\bibliography
