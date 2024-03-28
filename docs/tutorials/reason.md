# Detecting Reason of Hospitalisation

In this tutorial we will use the pipe `eds.reason` to :

- Identify spans that corresponds to the reason of hospitalisation
- Check if there are named entities overlapping with my span of 'reason of hospitalisation'
- Check for all named entities if they are tagged `is_reason`

```python
import edsnlp, edsnlp.pipes as eds

text = """COMPTE RENDU D'HOSPITALISATION du 11/07/2018 au 12/07/2018
MOTIF D'HOSPITALISATION
Monsieur Dupont Jean Michel, de sexe masculin, âgée de 39 ans, née le 23/11/1978,
a été hospitalisé du 11/08/2019 au 17/08/2019 pour attaque d'asthme.

ANTÉCÉDENTS
Antécédents médicaux :
Premier épisode d'asthme en mai 2018."""

nlp = edsnlp.blank("eds")

# Extraction d'entités nommées
nlp.add_pipe(
    eds.matcher(
        terms=dict(
            respiratoire=[
                "asthmatique",
                "asthme",
                "toux",
            ]
        ),
    ),
)
nlp.add_pipe(eds.normalizer())
nlp.add_pipe(eds.sections())
nlp.add_pipe(eds.reason(use_sections=True))

doc = nlp(text)
```

The pipe `reason` will add a key of spans called `reasons`. We check the first item in this list.

```python
# ↑ Omitted code above ↑

reason = doc.spans["reasons"][0]
reason
# Out: hospitalisé du 11/08/2019 au 17/08/2019 pour attaque d'asthme.
```

Naturally, all spans included the `reasons` key have the attribute `#!python reason._.is_reason == True`.

```python
# ↑ Omitted code above ↑

reason._.is_reason
# Out: True
```

<!-- no-check -->

```python
# ↑ Omitted code above ↑

entities = reason._.ents_reason  # (1)
for e in entities:
    print(
        "Entity:",
        e.text,
        "-- Label:",
        e.label_,
        "-- is_reason:",
        e._.is_reason,
    )
# Out: Entity: asthme -- Label: respiratoire -- is_reason: True
```

1. We check if the span include named entities, their labels and the attribute is_reason

We can verify that named entities that do not overlap with the spans of reason, have their attribute `#!python reason._.is_reason == False`:

```{ .python .no-check }
for e in doc.ents:
    print(e.start, e, e._.is_reason)
# Out: 42 asthme True
# Out: 54 asthme False
```
