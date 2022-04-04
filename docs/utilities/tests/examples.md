# Creating Examples

Testing a NER/qualifier pipeline can be a hassle. We created a utility to simplify that process.

Using the [`parse_example`][edsnlp.utils.examples.parse_example] method, you can define a full example in a human-readable way:

```python
from edsnlp.utils.examples import parse_example

example = "Absence d'<ent negated=true>image osseuse d'allure évolutive</ent>."

text, entities = parse_example(example)

text
# Out: "Absence d'image osseuse d'allure évolutive."

entities
# Out: [Entity(start_char=10, end_char=42, modifiers=[Modifier(key='negated', value=True)])]
```

Entities are defined using the `<ent>` tag. You can encode complexe information by adding keys into the tag (see example above). The `parse_example` method strips the text of the tags, and outputs a list of `Entity` objects that contain:

- the character indices of the entity ;
- custom user-defined "modifiers".

See the [dedicated reference page][edsnlp.utils.examples.parse_example] for more information.
