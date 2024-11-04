# Visualization

Let's see how to display the output of a pipeline on a single text.

```python
import edsnlp, edsnlp.pipes as eds

nlp = edsnlp.blank("eds")
nlp.add_pipe(eds.normalizer())
nlp.add_pipe(eds.sentences())
nlp.add_pipe(eds.covid())
nlp.add_pipe(eds.negation())
nlp.add_pipe(eds.hypothesis())
nlp.add_pipe(eds.family())

txt = "Le patient a le covid."
```

## Visualize entities in a document

To print a text and highlight the entities in it, you can use `spacy.displacy`.

```{ .python .no-check }
from spacy import displacy

doc = nlp(txt)
displacy.render(doc, style="ent")
```

will render like this:

<center>
<div class="entities" style="line-height: 2.5; direction: ltr">Le patient a le
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    covid
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">covid</span>
</mark>
.</div>
</center>

## Visualize entities as a table

To quickly visualize the output of a pipeline on a document, including the annotated extensions/qualifiers, you can convert the output to a DataFrame and display it.

```{ .python .no-check }
nlp.pipe([txt]).to_pandas(
    converter="ents",
    # Add any extension you want to display
    span_attributes=["negation", "hypothesis", "family"],
    # Shows the entities in doc.ents by default
    # span_getter=["ents"]
)
```

<div class="md-typeset">
<div class="md-typeset__table compact-table">

<table>
  <thead>
    <tr style="text-align: right;">
      <th>note_id</th>
      <th>start</th>
      <th>end</th>
      <th>label</th>
      <th>lexical_variant</th>
      <th>span_type</th>
      <th>negation</th>
      <th>hypothesis</th>
      <th>family</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>None</td>
      <td>16</td>
      <td>21</td>
      <td>covid</td>
      <td>covid</td>
      <td>ents</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

</div>
</div>
