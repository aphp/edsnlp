# Sections

Detected sections are :

- allergies ;
- antécédents ;
- antécédents familiaux ;
- traitements entrée ;
- conclusion ;
- conclusion entrée ;
- habitus ;
- correspondants ;
- diagnostic ;
- données biométriques entrée ;
- examens ;
- examens complémentaires ;
- facteurs de risques ;
- histoire de la maladie ;
- actes ;
- motif ;
- prescriptions ;
- traitements sortie.

It works by extracting section titles. Then, "sections" cover the entire text that is between two section titles (or the last title and the end of the document).

```{eval-rst}

.. warning::

    Use at your own risks : should you rely on `sections` for critical downstream tasks, you should validate the pipeline to make sure that the component works. For instance, the `antecedents` pipeline can use sections to make its predictions, but that possibility is deactivated by default.
```

## Declared extensions

The `sections` pipeline declares two [Spacy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), `Doc` objects :

1. The `section_titles` attribute is a list containing all section titles extracted using the list declared in the `terms.py` module.
2. The `sections` attribute contains a list of sections, ie spans of text between two section title (or the last title and the end of the document).

## Authors and citation

The `negation` pipeline was developed by the Data Science team at EDS. It uses [Ivan Lerner's dataset](https://gitlab.eds.aphp.fr/IvanL/section_dataset) of annotated section titles. It was reviewed by Gilles Chatellier, who grouped section titles by type.
