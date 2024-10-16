The components were developed by AP-HP's Data Science team with a team of medical experts, following the insights of the algorithm proposed by [@petitjean_2024]

Some general considerations about those components:

- Extracted entities are stored in `doc.ents` and `doc.spans`. For instance, the `eds.tobacco` component stores matches in `doc.spans["tobacco"]`.
- The matched comorbidity is also available under the `ent.label_` of each match.
- Matches have an associated `_.status` attribute taking the value `1`, or `2`. A corresponding `_.detailed_status` attribute stores the human-readable status, which can be component-dependent. See each component documentation for more details.
- Some components add additional information to matches. For instance, the `tobacco` adds, if relevant, extracted *pack-year* (= *paquet-année*). Those information are available under the `ent._.assigned` attribute.
- Those components work on **normalized** documents. Please use the `eds.normalizer` pipeline (see [Usage](#usage) below)

--8<-- "docs/pipes/ner/disorders/warning.md"

!!! warning "Use qualifiers"
    Those components **should be used with a qualification pipeline** to avoid extracted unwanted matches. At the very least, you should use available rule-based qualifiers (`eds.negation`, `eds.hypothesis` and `eds.family`). Better, a machine learning qualification component was developed and trained specifically for those components. For privacy reason, the model isn't publicly available yet.

    !!! aphp "Use the ML model"

        For projects working on AP-HP's CDW, this model is available via its models catalogue.

## Usage

```{ .python .no-check }
import edsnlp, edsnlp.pipes as eds

nlp = edsnlp.blank("eds")
nlp.add_pipe(eds.sentences())
nlp.add_pipe(
    eds.normalizer(
        accents=True,
        lowercase=True,
        quotes=True,
        spaces=True,
        pollution=dict(
            biology=True, #(1)
            coding=True, #(2)
        ),
    ),
)
nlp.add_pipe(eds.tobacco())
nlp.add_pipe(eds.diabetes())

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

tobacco_matches = doc.spans["tobacco"]
tobacco_matches[0]._.detailed_status
# Out: "ABSTINENCE" #

tobacco_matches[0]._.assigned["PA"]  # paquet-année
# Out: 10 # (3)


diabetes = doc.spans["diabetes"]
(diabetes[0]._.detailed_status, diabetes[1]._.detailed_status)
# Out: ('WITH_COMPLICATION', 'WITHOUT_COMPLICATION') # (4)
```

1. This will discard mentions of biology results, which often leads to false positive
2. This will discard mentions of ICD10 coding that sometimes appears at the end of clinical documents
3. Here we see an example of additional information that can be extracted
4. Here we see the importance of document-level aggregation to extract the correct severity of each comorbidity.
