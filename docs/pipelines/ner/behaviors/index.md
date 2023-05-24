# Behaviors

## Presentation

At the moment, EDS-NLP exposes two pipelines extracting behavioral patterns, namely the tobacco and alcohol consumption status. Each component is based on the ContextualMatcher component.
Some general considerations about those components:

- Extracted entities are stored in the `doc.spans` dictionary. For instance, the `eds.tobacco` component stores matches in `doc.spans["tobacco"]`.
- The comorbidity is also available under the `ent.label_` of each match.
- Matches have an associated `_.status` attribute taking the value `0`, `1`, or `2`. A corresponding `_.detailled_status` attribute stores the human-readable status, which can be component-dependent. See each component documentation for more details.
- Some components add additional information to matches. For instance, the `tobacco` adds, if relevant, extracted *pack-year* (= *paquet-année*). Those information are available under the `ent._.assigned` attribute.
- Those components work on **normalized** documents. Please use the `eds.normalizer` pipeline with the following parameters:
  <!-- no-check -->
  ```python
  nlp.add_pipe(
      "eds.normalizer",
      config=dict(
          accents=True,
          lowercase=True,
          quotes=True,
          spaces=True,
          pollution=dict(
              information=True,
              bars=True,
              biology=True,
              doctors=True,
              web=True,
              coding=True,
              footer=True,
          ),
      ),
  )
  ```

- Those components **should be used with a qualification pipeline** to avoid extracted unwanted matches. At the very least, you can use available rule-based qualifiers (`eds.negation`, `eds.hypothesis` and `eds.family`). Better, a machine learning qualification component was developped and trained specificaly for those components. For privacy reason, the model isn't publicly available yet.

    !!! aphp "Use the ML model"

        The model will soon be available in the models catalogue of AP-HP's CDW.

## Usage

<!-- no-check -->

```python
import spacy

nlp = spacy.blank("eds")
nlp.add_pipe("eds.sentences")
nlp.add_pipe(
    "eds.normalizer",
    config=dict(
        accents=True,
        lowercase=True,
        quotes=True,
        spaces=True,
        pollution=dict(
            information=True,
            bars=True,
            biology=True,
            doctors=True,
            web=True,
            coding=True,
            footer=True,
        ),
    ),
)
nlp.add_pipe("eds.tobacco")
nlp.add_pipe("eds.diabetes")

text = """
Compte-rendu de consultation.

Je vois ce jour M. SCOTT pour le suivi de sa rétinopathie diabétique.
Le patient va bien depuis la dernière fois.
Je le félicite pour la poursuite de son sevrage tabagique (toujours à 10 paquet-année).

Sur le plan de son diabète, la glycémie est stable.
"""

doc = nlp(text)

doc.spans
# Out: {
# 'pollutions': [],
# 'tobacco': [sevrage tabagique (toujours à 10 paquet-année],
# 'diabetes': [rétinopathie diabétique, diabète]
# }

tobacco = doc.spans["tobacco"]
tobacco[0]._.detailled_status
# Out: "ABSTINENCE" #

tobacco[0]._.assigned["PA"]  # paquet-année
# Out: 10 # (1)


diabetes = doc.spans["diabetes"]
(diabetes[0]._.detailled_status, diabetes[1]._.detailled_status)
# Out: ('WITH_COMPLICATION', 'WITHOUT_COMPLICATION') # (2)
```

1. Here we see an example of additional information that can be extracted
2. Here we see the importance of document-level aggregation to extract the correct severity of each comorbidity.
