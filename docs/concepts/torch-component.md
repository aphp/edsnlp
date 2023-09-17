# Torch Component {: #edsnlp.core.torch_component.TorchComponent }

Torch components allow for deep learning operations to be performed on the [Doc](https://spacy.io/api/doc) object and must be trained to be used. Such pipes can be used to train a model to detect named entities, predict the label of a document or an attribute of a text span, and so on.

<figure style="text-align: center" markdown="1">
![Sharing and nesting components](/assets/images/sharing-components.png){: style="height:230px" }
<figcaption>Example of sharing and nesting components</figcaption>
</figure>

## Anatomy of a trainable pipe

Building and running deep learning models usually requires to `preprocess` the input sample into features, to batch or `collate` these features together to process multiple samples at once, running deep learning operations over these features (in Pytorch, this step is done in the `forward` method) and to `postprocess` the outputs of these operation to complete the original sample.

In the trainable pipes of EDS-NLP, preprocessing and postprocessing are decoupled from the deep learning code but collocated with the forward method. This is achieved by splitting the class of a trainable component into four methods, which allows us to keep the development of new deep-learning components simple while ensuring efficient models both during training and inference.


/// details | Methods of a trainable component

### `preprocess` {: #edsnlp.core.torch_component.TorchComponent.preprocess }

::: edsnlp.core.torch_component.TorchComponent.preprocess
    options:
        heading_level: 4
        show_source: false
        show_toc: false

### `collate` {: #edsnlp.core.torch_component.TorchComponent.collate }

::: edsnlp.core.torch_component.TorchComponent.collate
    options:
        heading_level: 4
        show_source: false
        show_toc: false

### `forward` {: #edsnlp.core.torch_component.TorchComponent.forward }

::: edsnlp.core.torch_component.TorchComponent.forward
    options:
        heading_level: 4
        show_source: false
        show_toc: false

### `postprocess` {: #edsnlp.core.torch_component.TorchComponent.postprocess }

::: edsnlp.core.torch_component.TorchComponent.postprocess
    options:
        heading_level: 4
        show_source: false
        show_toc: false


Additionally, there is a fifth method:


### `post_init` {: #edsnlp.core.torch_component.TorchComponent.post_init }

::: edsnlp.core.torch_component.TorchComponent.post_init
    options:
        heading_level: 3
        show_source: false

///

## Nesting trainable pipes

Like pytorch modules, you can compose trainable pipes together to build complex architectures. For instance, a trainable named entity recognition component may delegate some of its logic to an embedding component, which will only be responsible for converting words into a embeddings. Nesting pipes allows switching parts of the neural networks to test various architectures and keeping the modelling logic modular.

!!! success "Nested preprocessing"

    This is where the idea of collocating `preprocess` and `collate` with `forward` really shines: much like pytorch modules, they can be called recursively on the subcomponents of a trainable pipe. This allows to extend the composition pattern to the preprocessing step to enable true modularity.


## Sharing subcomponents

Sharing parts of a neural network while training on different tasks can be an effective way to improve the network efficiency. For instance, it is common to share an embedding layer between multiple tasks that require embedding the same inputs.

In EDS-NLP, sharing a subcomponent is simply done by sharing the object between the multiple pipes. You can either refer to an existing subcomponent when configuring a new component in Python, or use the interpolation mechanism of our configuration system.

=== "API-based"

    ```{ .python .no-check }
    nlp.add_pipe(
        "my-component-1",
        name="first",
        config={
            "embedding": {
                "@factory": "eds.embeddings",
                # ...
            }
        },
    )
    nlp.add_pipe(
        "my-component-2",
        name="second",
        config={
            "embedding": pipeline.components.first.embedding,
        },
    )
    ```

=== "Configuration-based"

    ```toml
    [components.first]
    @factory = "my-component-1"

    [components.first.embedding]
    @factory = "eds.embeddings"
    ...

    [components.second]
    @factory = "my-component-2"
    embedding = ${components.first.embedding}
    ```

To avoid recomputing the `preprocess` / `forward` and `collate` in the multiple components that use it, we rely on a light cache system.

During the training loop, when computing the loss for each component, the forward calls must be wrapped by the [`pipeline.cache()`][edsnlp.core.pipeline.Pipeline.cache] context to enable this caching mechanism between components.


## Implementation example

Here is an example of a trainable component:

```python
from typing import Any, Dict, Iterable, Sequence

import torch
from tqdm import tqdm

from edsnlp import Pipeline, registry
from edsnlp.core.torch_component import TorchComponent
from spacy.tokens import Doc


@registry.factory.register("my-component")
class MyComponent(TorchComponent):
    def __init__(
        self,  # A subcomponent
        pipeline: Pipeline,
        name: str,
        embedding: TorchComponent,
    ):
        super().__init__(pipeline=pipeline, name=name)
        self.embedding = embedding

    def post_init(self, gold_data: Iterable[Doc], exclude: set):
        super().post_init(gold_data, exclude)

        # Initialize the component with the gold documents
        with self.label_vocabulary.initialization():
            for doc in tqdm(gold_data, desc="Initializing the component"):
                # Do something like learning a vocabulary over the initialization
                # documents
                ...

        # And post_init the subcomponent
        self.embedding.post_init(gold_data, exclude)

        # Initialize any layer that might be missing from the module
        self.classifier = torch.nn.Linear(...)

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        # Preprocess the doc to extract features required to run the embedding
        # subcomponent, and this component
        return {
            "embedding": self.embedding.preprocess(doc),
            "my-feature": ...(doc),
        }

    def collate(self, batch) -> Dict:
        # Collate the features of the "embedding" subcomponent
        # and the features of this component as well
        return {
            "embedding": self.embedding.collate(batch["embedding"]),
            "my-feature": torch.as_tensor(batch["my-feature"]),
        }

    def forward(self, batch: Dict) -> Dict:
        # Call the embedding subcomponent
        embeds = self.embedding(batch["embedding"])

        # Do something with the embedding tensors
        output = ...(embeds)

        return output

    def postprocess(self, docs: Sequence[Doc], output: Dict) -> Sequence[Doc]:
        # Annotate the docs with the outputs of the forward method
        ...
        return docs
```
