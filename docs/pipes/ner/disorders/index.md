# Disorders

## Presentation

The following components extract 16 different conditions from the [Charlson Comorbidity Index](https://www.rdplf.org/calculateurs/pages/charlson/charlson.html). Each component is based on the ContextualMatcher component.
Some general considerations about those components:

- Extracted entities are stored in `doc.ents` and `doc.spans`. For instance, the `eds.tobacco` component stores matches in `doc.spans["tobacco"]`.
- The matched comorbidity is also available under the `ent.label_` of each match.
- Matches have an associated `_.status` attribute taking the value `0`, `1`, or `2`. A corresponding `_.detailed_status` attribute stores the human-readable status, which can be component-dependent. See each component documentation for more details.
- Some components add additional information to matches. For instance, the `tobacco` adds, if relevant, extracted *pack-year* (= *paquet-année*). Those information are available under the `ent._.assigned` attribute.
- Those components work on **normalized** documents. Please use the `eds.normalizer` pipeline with the following parameters:

    ```{ .python .no-check }
    import edsnlp, edsnlp.pipes as eds
    ...

    nlp.add_pipe(
        eds.normalizer(
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

- Those components **should be used with a qualification pipeline** to avoid extracted unwanted matches. At the very least, you can use available rule-based qualifiers (`eds.negation`, `eds.hypothesis` and `eds.family`). Better, a machine learning qualification component was developed and trained specifically for those components. For privacy reason, the model isn't publicly available yet.

    !!! aphp "Use the ML model"

        The model will soon be available in the models catalogue of AP-HP's CDW.

!!! tip "On the medical definition of the comorbidities"

    Those components were developped to extract **chronic** and **symptomatic** conditions only.

## Aggregation

For relevant phenotyping, matches should be aggregated at the document-level. For instance, a document might mention a complicated diabetes at the beginning ("*Le patient a une rétinopathie diabétique*"), and then refer to this diabetes without mentionning that it is complicated anymore ("*Concernant son diabète, le patient ...*").
Thus, a good and simple aggregation rule is, for each comorbidity, to

- disregard all entities tagged as irrelevant by the qualification component(s)
- take the maximum (i.e., the most severe) status of the leftover entities

An implementation of this rule is presented [here][aggregating-results]
