# Writing a training script

In this tutorial, we'll see how we can write our own deep learning model training script with EDS-NLP. We will implement a script to train a named-entity recognition (NER) model.

If you do not care about the details and just want to train a model, we suggest that you use the [training API](/concepts/deep-learning) and move on to the [next tutorial](/tutorials/training-ner).

!!! warning "Hardware requirements"

    Training a modern deep learning model requires a lot of computational resources. We recommend using a machine with a GPU, ideally with at least 16GB of VRAM. If you don't have access to a GPU, you can use a cloud service like [Google Colab](https://colab.research.google.com/), [Kaggle](https://www.kaggle.com/), [Paperspace](https://www.paperspace.com/) or [Vast.ai](https://vast.ai/).

Under the hood, EDS-NLP uses PyTorch to train deep-learning models. EDS-NLP acts as a sidekick to PyTorch, providing a set of tools to perform preprocessing, composition and evaluation. The trainable [`TorchComponents`][edsnlp.core.torch_component.TorchComponent] are actually PyTorch modules with a few extra methods to handle the feature preprocessing and postprocessing. Therefore, EDS-NLP is fully compatible with the PyTorch ecosystem.

## Step-by-step walkthrough

Training a supervised deep-learning model consists in feeding batches of annotated samples taken from a training corpus to a model and optimizing its parameters of the model to decrease its prediction
error. The process of training a pipeline with EDS-NLP is structured as follows:

### 1. Defining the model

We first start by seeding the random states and instantiating a new trainable pipeline composed of [trainable pipes](/pipes/trainable). The model described here computes text embeddings with a pre-trained transformer followed by a CNN, and performs
the NER prediction task using a Conditional Random Field (CRF) token classifier.

```python
import edsnlp, edsnlp.pipes as eds
from confit.utils.random import set_seed

set_seed(42)

nlp = edsnlp.blank("eds")
nlp.add_pipe(
    eds.ner_crf(  # (1)!
        mode="joint",  # (2)!
        target_span_getter="gold-ner",  # (3)!
        window=20,
        embedding=eds.text_cnn(  # (4)!
            kernel_sizes=[3],
            embedding=eds.transformer(  # (5)!
                model="prajjwal1/bert-tiny",  # (6)!
                window=128,
                stride=96,
            ),
        ),
    ),
    name="ner",
)
```

1. We use the `eds.ner_crf` NER task module, which classifies word embeddings into NER labels (BIOUL scheme) using a CRF.
2. Each component of the pipeline can be configured with a dictionary, using the parameter described in the component's page.
3. The `target_span_getter` parameter defines the name of the span group used to train the NER model. In this case, the model will look for the entities to train on in `doc.spans["gold-ner"]`. This is important because we might store entities in other span groups with a different purpose (e.g. `doc.spans["sections"]` contain the sections Spans, but we don't want to train on these). We will need to make sure the entities from the training dataset are assigned to this span group (next section).
4. The word embeddings used by the CRF are computed by a CNN, which builds on top of another embedding layer.
5. The base embedding layer is a pretrained transformer, which computes contextualized word embeddings.
6. We chose the `prajjwal1/bert-tiny` model in this tutorial for testing purposes, but we recommend using a larger model like `bert-base-cased` or `camembert-base` (French) for real-world applications.

### 2. Loading the raw dataset and convert it into Doc objects

To train a pipeline, we must convert our annotated data into `Doc` objects that will be either used as training samples or evaluation samples. We will assume the dataset is in [Standoff format](/data/standoff), usually produced by the [Brat](https://brat.nlplab.org) annotation tool, but any format can be used.

At this step, we might also want to perform data augmentation, filtering, splitting or any other data transformation. In this tutorial, we will split on line jumps and filter out empty documents from the training data. We will use our [Stream][edsnlp.core.stream.Stream] API to handle the data processing, but you can use any method you like, so long as you end up with a collection of `Doc` objects.

```{ .python .no-check }
import edsnlp


def skip_empty_docs(batch):
    for doc in batch:
        if len(doc.ents) > 0:
            yield doc


training_data = (
    edsnlp.data.read_standoff(  # (1)!
        train_data_path,
        tokenizer=nlp.tokenizer,  # (2)!
        span_setter=["ents", "gold-ner"],  # (3)!
    )
    .map(eds.split(regex="\n\n"))  # (4)!
    .map_batches(skip_empty_docs)  # (5)!
)
```

1. Read the data from the brat directory and convert it into Docs.
2. Tokenize the training docs with the same tokenizer as the trained model
3. Store the annotated Brat entities as spans in `doc.ents`, and `doc.spans["gold-ner"]`
4. Split the documents on line jumps.
5. Filter out empty documents.

As for the validation data, we will keep all the documents, even empty ones, to obtain representative metrics.

```{ .python .no-check }
val_data = edsnlp.data.read_standoff(
    val_data_path,
    tokenizer=nlp.tokenizer,
    span_setter=["ents", "gold-ner"],
)
val_docs = list(val_data)  # (1)!
```

1. Cache the stream result into a list of `Doc`

### 3. Complete the initialization of the model

We initialize the missing or incomplete components attributes (such as label vocabularies) with the training dataset. Indeed, when defining the model, we specified the architecture of the model, but we did not specify the types of named entities that the model will predict. This can be done either

- explicitly by setting the `labels` parameter in `eds.ner_crf` in the [definition](#1-defining-the-model) above,
- automatically with `post_init`: then `eds.ner_crf` looks in `doc.spans[target_span_getter]` of all docs in `training_data` to infer the labels.

```{ .python .no-check }
nlp.post_init(training_data)
```

### 4. Making the stream of mini-batches

The training dataset of `Doc` objects is then preprocessed into features to be fed to the model during the training loop. We will continue to use EDS-NLP's streams to handle the data processing :

- We first request the training data stream to loop on the input data, since we want that each example is seen multiple times during the training until a given number of steps is reached

    ??? note "Looping in EDS-NLP Streams"

        Note that in EDS-NLP, looping on a stream is always done on the input data, no matter when `loop()` is called. This means that shuffling or any further preprocessing step will be applied multiple times, each time we loop. This is usually a good thing if preprocessing contains randomness to increase the diversity of the training samples while avoiding loading multiple versions of a same document in memory. To loop after preprocessing, we can collect the stream into a list and loop on the list (`edsnlp.data.from_iterable(list(training_data)), loop=True`).

- We shuffle the data before batching to diversify the samples in each mini-batch
- We extract the features and labels required by each component (and sub-components) of the pipeline
- Finally, we group the samples into mini-batches, such that each mini-batch contains a maximum number of tokens, or any other batching criterion and assemble (or "collate") the features into tensors

```{ .python .no-check }
from edsnlp.utils.batching import stat_batchify

device = "cuda" if torch.cuda.is_available() else "cpu"  # (1)!
batches = (
   training_data.loop()
    .shuffle("dataset")  # (2)!
    .map(nlp.preprocess, kwargs={"supervision": True})  # (3)!
    .batchify(batch_size=32 * 128, batch_by=stat_batchify("tokens"))  # (4)!
    .map(nlp.collate, kwargs={"device": device})
)
```

1. Check if a GPU is available and set the device accordingly.
2. Apply shuffling to our stream. If our dataset is too large to fit in memory, instead of "dataset" we can set the shuffle batch size to "100 docs" for example, or "fragment" for parquet datasets.
3. This will call the `preprocess_supervised` method of the [TorchComponent][edsnlp.core.torch_component.TorchComponent] class and return a nested dictionary containing the required features and labels.
4. Make batches that contain at most 32 * 128 tokens (e.g. 32 samples of 128 tokens, but this accounts samples may have different lengths). We use the `stat_batchify` function to look for a key containing `tokens` in the features `stats` sub-dictionary and add samples to the batch until the sum of the `*tokens*` stats exceeds 32 * 128.


and that's it ! We now have a looping stream of mini-batches that we can feed to our model.
For better efficiency, we can also perform this in parallel in a separate worker by setting `num_cpu_workers` to 1 or more.
Note that streams in EDS-NLP are lazy, meaning that the execution has not started yet, and the data is not loaded in memory. This will only happen when we start iterating over the stream in the next section.

```{ .python .no-check }
batches = batches.set_processing(
   num_cpu_workers=1,
   process_start_method="spawn"  # (1)!
)
```

1. Since we use a GPU, we must use the "spawn" method to create the workers. This is because the default multiprocessing "fork" method is not compatible with CUDA.

### 5. The training loop

We instantiate a pytorch optimizer and start the training loop

```{ .python .no-check }
from itertools import chain, repeat
from tqdm import tqdm
import torch

lr = 3e-4
max_steps = 400

# Move the model to the GPU
nlp.to(device)

optimizer = torch.optim.AdamW(
    params=nlp.parameters(),
    lr=lr,
)

iterator = iter(batches)

for step in tqdm(range(max_steps), "Training model", leave=True):
    batch = next(iterator)
    optimizer.zero_grad()
```

### 6. Optimizing the weights

Inside the training loop, the trainable components are fed the collated batches from the dataloader by calling the [`TorchComponent.forward`][edsnlp.core.torch_component.TorchComponent.forward] method (via a simple call) to compute the losses. In the case we train a multitask model (not in this tutorial) and the outputs of a shared embedding are reused between components, we enable caching by wrapping this step in a cache context. The training loop is otherwise carried in a similar fashion to a standard pytorch training loop.

```{ .python .no-check }
    with nlp.cache():
        loss = torch.zeros((), device=device)
        for name, component in nlp.torch_components():
            output = component(batch[name])
            if "loss" in output:
                loss += output["loss"]

    loss.backward()

    optimizer.step()
```

### 7. Evaluating the model

Finally, the model is evaluated on the validation dataset and saved at regular intervals. We will use the `NerExactMetric` to evaluate the NER performance using Precision, Recall and F1 scores. This metric only counts an entity as correct if it matches the label and boundaries of a target entity.

```{ .python .no-check }
from edsnlp.metrics.ner import NerExactMetric
from copy import deepcopy

metric = NerExactMetric(span_getter=nlp.pipes.ner.target_span_getter)

    ...
    if ((step + 1) % 100) == 0:
        with nlp.select_pipes(enable=["ner"]):  # (1)!
            preds = deepcopy(val_docs)
            for doc in preds:
                doc.ents = doc.spans["gold-ner"] = []  # (2)!
            preds = nlp.pipe(preds)  # (3)!
            print(metric(val_docs, preds))

    nlp.to_disk("model")  #(4)!
```

1. In the case we have multiple pipes in our model, we may want to selectively evaluate each pipe, thus we use the `select_pipes` method to disable every pipe except "ner".
2. Clean the documents that our model will annotate
3. We use the `pipe` method to run the "ner" component on the validation dataset. This method is similar to the `__call__` method of EDS-NLP components, but it is used to run a component on a list of
   Docs. This is also equivalent to
    ```{ .python .no-check }
    preds = (
        edsnlp.data
       .from_iterable(preds)
       .map_pipeline(nlp)
    )
    ```
4. We could also have saved the model with `torch.save(model, "model.pt")`, but `nlp.to_disk` avoids pickling and allows to inspect the model's files by saving them into a structured directory.

## Full example

Let's wrap the training code in a function, and make it callable from the command line using [confit](https://github.com/aphp/confit) !

??? example "train.py"

    ```python linenums="1"
    from copy import deepcopy
    from typing import Iterator

    import torch
    from confit import Cli
    from tqdm import tqdm

    import edsnlp
    import edsnlp.pipes as eds
    from edsnlp.metrics.ner import NerExactMetric
    from edsnlp.utils.batching import stat_batchify

    app = Cli(pretty_exceptions_show_locals=False)


    @app.command(name="train", registry=edsnlp.registry)  # (1)!
    def train_model(
        nlp: edsnlp.Pipeline,
        train_data_path: str,
        val_data_path: str,
        batch_size: int = 32 * 128,
        lr: float = 1e-4,
        max_steps: int = 400,
        num_preprocessing_workers: int = 1,
        evaluation_interval: int = 100,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define function to skip empty docs
        def skip_empty_docs(batch: Iterator) -> Iterator:
            for doc in batch:
                if len(doc.ents) > 0:
                    yield doc

        # Load and process training data
        training_data = (
            edsnlp.data.read_standoff(
                train_data_path,
                span_setter=["ents", "gold-ner"],
                tokenizer=nlp.tokenizer,
            )
            .map(eds.split(regex="\n\n"))
            .map_batches(skip_empty_docs)
        )

        # Load validation data
        val_data = edsnlp.data.read_standoff(
            val_data_path,
            span_setter=["ents", "gold-ner"],
            tokenizer=nlp.tokenizer,
        )
        val_docs = list(val_data)

        # Initialize components
        nlp.post_init(training_data)

        # Prepare the stream of batches
        batches = (
            training_data.loop()
            .shuffle("dataset")
            .map(nlp.preprocess, kwargs={"supervision": True})
            .batchify(batch_size=batch_size, batch_by=stat_batchify("tokens"))
            .map(nlp.collate, kwargs={"device": device})
            .set_processing(num_cpu_workers=1, process_start_method="spawn")
        )

        # Move the model to the GPU if available
        nlp.to(device)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(params=nlp.parameters(), lr=lr)

        metric = NerExactMetric(span_getter=nlp.pipes.ner.target_span_getter)

        # Training loop
        iterator = iter(batches)
        for step in tqdm(range(max_steps), "Training model", leave=True):
            batch = next(iterator)
            optimizer.zero_grad()

            with nlp.cache():
                loss = torch.zeros((), device=device)
                for name, component in nlp.torch_components():
                    output = component(batch[name])
                    if "loss" in output:
                        loss += output["loss"]

            loss.backward()
            optimizer.step()

            # Evaluation and model saving
            if ((step + 1) % evaluation_interval) == 0:
                with nlp.select_pipes(enable=["ner"]):
                    # Clean the documents that our model will annotate
                    preds = deepcopy(val_docs)
                    for doc in preds:
                        doc.ents = doc.spans["gold-ner"] = []
                    preds = nlp.pipe(preds)
                    print(metric(val_docs, preds))

                nlp.to_disk("model")


    if __name__ == "__main__":
        nlp = edsnlp.blank("eds")
        nlp.add_pipe(
            eds.ner_crf(
                mode="joint",
                target_span_getter="gold-ner",
                window=20,
                embedding=eds.text_cnn(
                    kernel_sizes=[3],
                    embedding=eds.transformer(
                        model="prajjwal1/bert-tiny",
                        window=128,
                        stride=96,
                    ),
                ),
            ),
            name="ner",
        )
        train_model(
            nlp,
            train_data_path="my_brat_data/train",
            val_data_path="my_brat_data/val",
            batch_size=32 * 128,
            lr=1e-4,
            max_steps=1000,
            num_preprocessing_workers=1,
            evaluation_interval=100,
        )
    ```

    1. This will become useful in the next section, when we will use the configuration file to define the pipeline. If you don't want to use a configuration file, you can remove this decorator.

We can now copy the above code in a notebook and run it, or call this script from the command line:

```{: data-md-color-scheme="slate" }
python train.py
```

At the end of the training, the pipeline is ready to use since every trained component of the pipeline is self-sufficient, ie contains the preprocessing, inference and postprocessing code required to run it.

## Configuration

To decouple the configuration and the code of our training script, let's define a configuration file where we will describe **both** our training parameters and the pipeline. You can either write the
config of the pipeline by hand, or generate a pipeline config draft from an instantiated pipeline by running:

```{ .python .no-check }
print(nlp.config.to_yaml_str())
```

```yaml title="config.yml"
nlp:
  "@core": "pipeline"
  lang: "eds"
  components:
    ner:
      "@factory": "eds.ner_crf"
      mode: "joint"
      target_span_getter: "gold-ner"
      window: 20

      embedding:
        "@factory": "eds.text_cnn"
        kernel_sizes: [3]

        embedding:
          "@factory": "eds.transformer"
          model: "prajjwal1/bert-tiny"
          window: 128
          stride: 96

train:
  nlp: ${ nlp }
  train_data_path: my_brat_data/train
  val_data_path: my_brat_data/val
  batch_size: ${ 32 * 128 }
  lr: 1e-4
  max_steps: 400
  num_preprocessing_workers: 1
  evaluation_interval: 100
```

And replace the end of the script by

```{ .python .no-check }
if __name__ == "__main__":
    app.run()
```

That's it ! We can now call the training script with the configuration file as a parameter, and override some of its values:

```{: .shell data-md-color-scheme="slate" }
python train.py --config config.cfg --nlp.components.ner.embedding.embedding.transformer.window=64 --seed 43
```

## Going further

EDS-NLP also provides a generic training script that follows the same structure as the one we just wrote. You can learn more about in the [next NER model training tutorial through EDS-NLP training API](/tutorials/training-ner).

This tutorial gave you a glimpse of the training API of EDS-NLP. To build a custom trainable component, you can refer to the [TorchComponent][edsnlp.core.torch_component.TorchComponent] class or look up the implementation of [some of the trainable components on GitHub](https://github.com/aphp/edsnlp/tree/master/edsnlp/pipes/trainable).

We also recommend looking at an existing project as a reference, such as [eds-pseudo](https://github.com/aphp/eds-pseudo) or [mlg-norm](https://github.com/percevalw/mlg-norm).
