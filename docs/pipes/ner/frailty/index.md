# Frailty

## Presentation

The following components extract mentions of frailty across the different domains of the [Practical Geriatric Assessment (PGA)](https://ascopubs.org/doi/10.1200/JCO.23.00933), along with a couple domains that are not strictly speaking part of the PGA, but were considered relevant for frailty evaluation by the clinicians involved in the development process. Each component is based on the ContextualMatcher component.

Some general considerations about those components:

- Extracted entities are stored in `doc.ents` and `doc.spans`. For instance, the `eds.autonomy` component stores matches in `doc.spans["autonomy"]`.
- Matches have an associated `_.{domain}` attribute taking a value among  `healthy`, `altered_nondescript`, `altered_mild`, `altered_severe` and `other`, indicating the level of alteration of the match for the corresponding domain. Some terms may match for several frailty domains at once. For example, "EHPAD" will have both its `_.autonomy` and `_.social` statuses set to `altered_severe`.
- Some components are tailored to match some well-defined and standardized frailty scores, rather than the corresponding broader frailty domain. However, these components store their matches in `doc.spans` both under their own name and their corresponding domain name, and the matches will have attributes corresponding to the score itself, with the found value, and to the domain with the level of alteration. For example, a pipeline with the `eds.adl` component would match the phrase "ADL 6/6" and store it both in `doc.spans["adl"]` and `doc.spans["autonomy"]`. The match would have a `_.adl` attribute set to `6.0`, and a `_.autonomy` attribute set to `healthy`.
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

!!! warning "Use qualifiers"
    Those components **should be used with a qualification pipeline** to avoid extracted unwanted matches. At the very least, you can use available rule-based qualifiers (`eds.negation`, `eds.hypothesis` and `eds.family`). It is also possible to use a deep-learning-based qualifier model, but for privacy purposes, such models can't be shared publicly for now.


!!! tip "On the medical definition of frailty"

    Those components were developped to extract **chronic** alterations, and not **acute** events.

## Aggregation

For relevant phenotyping, matches should be aggregated at the document-level. For instance, a document containing an evaluation of the autonomy of a patient may have several terms matched by the `eds.autonomy` component, some of which of different levels of alteration.
Thus, a good and simple aggregation rule is, for each domain, to

- disregard all entities tagged as irrelevant by the qualification component(s)
- take the maximum (i.e., the most severe) status of the leftover entities
