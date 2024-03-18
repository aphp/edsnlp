# Making a training script

In this tutorial, we'll see how we can train a deep learning model with EDS-NLP. We will implement a script to train a named-entity recognition (NER) model.

## Step-by-step walkthrough

Training a supervised deep-learning model consists in feeding batches of annotated samples taken from a training corpus to a model and optimizing its parameters of the model to decrease its prediction
error. The process of training a pipeline with EDS-NLP is structured as follows:

### 1. Defining the model

We first start by seeding the random states and instantiating a new trainable pipeline. The model described here computes text embeddings with a pre-trained transformer followed by a CNN, and performs
the NER prediction task using a Conditional Random Field (CRF) token classifier. To compose deep-learning modules, we nest them in a dictionary : each new dictionary will instantiate a new module, and
the `@factory` key will be used to select the class of the module.

```{ .python .annotate }
import edsnlp
from confit.utils.random import set_seed

set_seed(42)

nlp = edsnlp.blank("eds")
nlp.add_pipe(
    "eds.ner_crf",  # (1)
    name="ner",
    config={
        "mode": "joint",  # (2)
        "target_span_getter": "ml-ner", # (3)
        "embedding": {
            "@factory": "eds.text_cnn",  # (4)
            "kernel_sizes": [3],
            "embedding": {
                "@factory": "eds.transformer",  # (5)
                "model": "prajjwal1/bert-tiny",  # (6)
                "window": 128,
                "stride": 96,
            },
        },
    },
)
```

1. We use the `eds.ner_crf` NER task module, which classifies word embeddings into NER labels (BIOUL scheme) using a CRF.
2. Each component of the pipeline can be configured with a dictionary, using the parameter described in the component's page.
3. The `target_span_getter` parameter defines the name of the span group used to train the NER model. We will need to make sure the entities from the training dataset are assigned to this span group (next section).
4. The word embeddings used by the CRF are computed by a CNN, which builds on top of another embedding layer.
5. The base embedding layer is a pretrained transformer, which computes contextualized word embeddings.
6. We chose the `prajjwal1/bert-tiny` model in this tutorial for testing purposes, but we recommend using a larger model like `bert-base-cased` or `camembert-base` (French) for real-world applications.

### 2. Adapting a dataset

To train a pipeline, we must convert our annotated data into documents that will be either used as training samples or a evaluation samples. This is done by designing a function to convert the dataset
into a list of spaCy Doc objects. We will assume the dataset has been annotated using [Brat](https://brat.nlplab.org), but any format can be used.

At this step, we might also want to perform data augmentation, filtering, splitting or any other data transformation. Note that this function will be used to load both the training data and the test
data.

```python
from pydantic import DirectoryPath
import edsnlp


@edsnlp.registry.adapters.register("ner_adapter")
def ner_adapter(
    path: DirectoryPath,
    skip_empty: bool = False,  # (1)
):
    def generator(nlp):
        # Read the data from the brat directory and convert it into Docs,
        docs = edsnlp.data.read_standoff(
            path,
            # Store spans in default "ents", and "ml-ner" for the training (prev. section)
            span_setter=["ents", "ml-ner"],
            # Tokenize the training docs with the same tokenizer as the trained model
            tokenizer=nlp.tokenizer,
        )
        for doc in docs:
            if skip_empty and len(doc.ents) == 0:
                continue
            yield doc

    return generator
```

1. We can skip documents that do not contain any annotations. However, this parameter should be false when loading documents used to evaluate the pipeline.

### 3. Loading the data

We then load and adapt (i.e., convert into spaCy Doc objects) the training and validation dataset. Since the adaption of raw documents depends on tokenization used in the trained model, we need to
pass the model to the adapter function.

```{ .python .no-check }
train_adapter = ner_adapter(train_data_path)
val_adapter = ner_adapter(val_data_path)

train_docs = list(train_adapter(nlp))
val_docs = list(val_adapter(nlp))
```

### 4. Complete the initialization with the training data

We initialize the missing or incomplete components attributes (such as label vocabularies) with the training dataset

```{ .python .no-check }
nlp.post_init(train_docs)
```

### 5. Preprocessing the data

The training dataset is then preprocessed into features. The resulting preprocessed dataset is then wrapped into a pytorch DataLoader to be fed to the model during the training loop with the model's
own collate method.

```{ .python .no-check }
import torch

batch_size = 8

preprocessed = list(
    nlp.preprocess_many(  # (1)
       train_docs,
       supervision=True,
    )
)
dataloader = torch.utils.data.DataLoader(
    preprocessed,
    batch_size=batch_size,
    collate_fn=nlp.collate,
    shuffle=True,
)
```

1. This will call the `preprocess_supervised` method of the [TorchComponent][edsnlp.core.torch_component.TorchComponent] class on every document and return a list of dictionaries containing the
   features and labels of each document.

### 6. Looping through the training data

We instantiate an optimizer and start the training loop

```{ .python .no-check }
from itertools import chain, repeat
from tqdm import tqdm

lr = 3e-4
max_steps = 400

optimizer = torch.optim.AdamW(
    params=nlp.parameters(),
    lr=lr,
)

# We will loop over the dataloader
iterator = chain.from_iterable(repeat(dataloader))

for step in tqdm(range(max_steps), "Training model", leave=True):
    batch = next(iterator)
    optimizer.zero_grad()
```

### 7. Optimizing the weights

Inside the training loop, the trainable components are fed the collated batches from the dataloader by calling
the [`TorchComponent.module_forward`][edsnlp.core.torch_component.TorchComponent.module_forward] methods to compute the losses. In the case we train a multi-task model (not in this tutorial), the
outputs of shared embedding are reused between components, we enable caching by wrapping this step in a cache context. The training loop is otherwise carried in a similar fashion to a standard pytorch
training loop

```{ .python .no-check }
    with nlp.cache():
        loss = torch.zeros((), device="cpu")
        for name, component in nlp.torch_components():
            output = component.module_forward(batch[name])  # (1)
            if "loss" in output:
                loss += output["loss"]

    loss.backward()

    optimizer.step()
```

1. We use the `module_forward` instead of a standard call, since the `__call__` method of EDS-NLP components is used to run a component on a spaCy Doc.

### 8. Evaluating the model

Finally, the model is evaluated on the validation dataset and saved at regular intervals.

```{ .python .no-check }
from edsnlp.scorers.ner import create_ner_exact_scorer
from copy import deepcopy

scorer = create_ner_exact_scorer(nlp.pipes.ner.target_span_getter)

    ...

    if (step % 100) == 0:
        with nlp.select_pipes(enable=["ner"]):  # (1)
            print(scorer(val_docs, nlp.pipe(deepcopy(val_docs))))  # (2)

    nlp.to_disk("model")  # (3)
```

1. In the case we have multiple pipes in our model, we may want to selectively evaluate each pipe, thus we use the `select_pipes` method to disable every pipe except "ner".
2. We use the `pipe` method to run the "ner" component on the validation dataset. This method is similar to the `__call__` method of EDS-NLP components, but it is used to run a component on a list of
   spaCy Docs.
3. We could also have saved the model with `torch.save(model, "model.pt")`, but `nlp.to_disk` avoids pickling and allows to inspect the model's files by saving them into a structured directory.

## Full example

Let's wrap the training code in a function, and make it callable from the command line using [confit](https://github.com/aphp/confit) !

??? example "train.py"

    ```python linenums="1"
    from copy import deepcopy
    from itertools import chain, repeat
    from typing import Callable, Iterable

    import torch
    from confit import Cli
    from pydantic import DirectoryPath
    from spacy.tokens import Doc
    from tqdm import tqdm

    import edsnlp
    from edsnlp import registry, Pipeline
    from edsnlp.scorers.ner import create_ner_exact_scorer


    @registry.adapters.register("ner_adapter")
    def ner_adapter(
        path: DirectoryPath,
        skip_empty: bool = False,
    ):
        def generator(nlp):
            # Read the data from the brat directory and convert it into Docs,
            docs = edsnlp.data.read_standoff(
               path,
               # Store spans in default "ents", and "ml-ner" for the training
               span_setter=["ents", "ml-ner"],
               # Tokenize the training docs with the same tokenizer as the trained model
               tokenizer=nlp.tokenizer,
            )
            for doc in docs:
                if skip_empty and len(doc.ents) == 0:
                    continue
                doc.spans["ml-ner"] = doc.ents
                yield doc

        return generator


    app = Cli(pretty_exceptions_show_locals=False)


    @app.command(name="train", registry=registry)  # (1)
    def train(
        nlp: Pipeline,
        train_adapter: Callable[[Pipeline], Iterable[Doc]],
        val_adapter: Callable[[Pipeline], Iterable[Doc]],
        max_steps: int = 1000,
        seed: int = 42,
        lr: float = 3e-4,
        batch_size: int = 4,
    ):
        # Adapting a dataset
        train_docs = list(train_adapter(nlp))
        val_docs = list(val_adapter(nlp))

        # Complete the initialization with the training data
        nlp.post_init(train_docs)

        # Preprocessing the data
        preprocessed = list(
            nlp.preprocess_many(
                train_docs,
                supervision=True,
            )
        )
        dataloader = torch.utils.data.DataLoader(
            preprocessed,
            batch_size=batch_size,
            collate_fn=nlp.collate,
            shuffle=True,
        )

        scorer = create_ner_exact_scorer(nlp.pipes.ner.target_span_getter)

        optimizer = torch.optim.AdamW(
            params=nlp.parameters(),
            lr=lr,
        )

        iterator = chain.from_iterable(repeat(dataloader))

        # Looping through the training data
        for step in tqdm(range(max_steps), "Training model", leave=True):
            batch = next(iterator)
            optimizer.zero_grad()

            loss = torch.zeros((), device="cpu")
            with nlp.cache():
                for name, component in nlp.torch_components():
                    output = component.module_forward(batch[name])
                    if "loss" in output:
                        loss += output["loss"]

            loss.backward()

            optimizer.step()

            # Evaluating the model
            if (step % 100) == 0:
                with nlp.select_pipes(enable=["ner"]):  #
                    print(scorer(val_docs, nlp.pipe(deepcopy(val_docs))))  #

            nlp.to_disk("model")


    if __name__ == "__main__":
        nlp = edsnlp.blank("eds")
        nlp.add_pipe(
            "eds.ner_crf",
            name="ner",
            config={
                "mode": "joint",
                "target_span_getter": "ml-ner",
                "window": 20,
                "embedding": {
                    "@factory": "eds.text_cnn",
                    "kernel_sizes": [3],
                    "embedding": {
                        "@factory": "eds.transformer",
                        "model": "prajjwal1/bert-tiny",
                        "window": 128,
                        "stride": 96,
                    },
                },
            },
        )
        train(
            nlp=nlp,
            train_adapter=ner_adapter("data/train"),
            val_adapter=ner_adapter("data/val"),
            max_steps=400,
            seed=42,
            lr=3e-4,
            batch_size=8,
        )
    ```

    1. This will become useful in the next section, when we will use the configuration file to define the pipeline. If you don't want to use a configuration file, you can remove this decorator.

We can now copy the above code in a notebook and run it, or call this script from the command line:

```{: data-md-color-scheme="slate" }
python train.py --seed 42
```

At the end of the training, the pipeline is ready to use (with the `.pipe` method) since every trained component of the pipeline is self-sufficient, ie contains the preprocessing, inference and
postprocessing code required to run it.

## Configuration

To decouple the configuration and the code of our training script, let's define a configuration file where we will describe **both** our training parameters and the pipeline. You can either write the
config of the pipeline by hand, or generate a pipeline config draft from an instantiated pipeline by running:

```{ .python .no-check }
print(nlp.config.to_str())
```

```toml title="config.cfg"
# This is this equivalent of the API-based declaration
# at the beginning of the tutorial
[nlp]
lang = "eds"
pipeline = ["ner"]
components = ${ components }

[components]

[components.ner]
@factory = "eds.ner_crf"
mode = "joint"
target_span_getter = "ml-ner"
window = 20
embedding = ${ cnn }

[cnn]
@factory = "eds.text_cnn"
kernel_sizes = [3]
embedding = ${ transformer }

[transformer]
@factory = "eds.transformer"
model = "prajjwal1/bert-tiny"
window = 128
stride = ${ transformer.window//2 }

# This is were we define the training script parameters
# the "train" section refers to the name of the command
# in the training script
[train]
nlp = ${ nlp }
train_adapter = { "@adapters": "ner_adapter", "path": "data/train" }
val_adapter = { "@adapters": "ner_adapter", "path": "data/val" }
max_steps = 400
seed = 42
lr = 3e-4
batch_size = 8
```

And replace the end of the script by

```{ .python .no-check }
if __name__ == "__main__":
    app.run()
```

That's it ! We can now call the training script with the configuration file as a parameter, and override some of its values:

```{: .shell data-md-color-scheme="slate" }
python train.py --config config.cfg --transformer.window=64 --seed 43
```

## Going further

This tutorial gave you a glimpse of the training API of EDS-NLP. We provide a more complete example of a training script in tests
at [tests/training/test_training.py](https://github.com/aphp/edsnlp/blob/master/tests/training/test_training.py). To build a custom trainable component, you can refer to
the [TorchComponent][edsnlp.core.torch_component.TorchComponent] class or look up the implementation of some of the trainable components on GitHub.
