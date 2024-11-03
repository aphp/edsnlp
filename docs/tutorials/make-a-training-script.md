# Custom training script

In this tutorial, we'll see how we can write our own deep learning model training script with EDS-NLP. We will implement a script to train a named-entity recognition (NER) model.

## Step-by-step walkthrough

Training a supervised deep-learning model consists in feeding batches of annotated samples taken from a training corpus to a model and optimizing its parameters of the model to decrease its prediction
error. The process of training a pipeline with EDS-NLP is structured as follows:

### 1. Defining the model

We first start by seeding the random states and instantiating a new trainable pipeline. The model described here computes text embeddings with a pre-trained transformer followed by a CNN, and performs
the NER prediction task using a Conditional Random Field (CRF) token classifier. To compose deep-learning modules, we simply compose them using the `eds.___` factories.

```python
import edsnlp, edsnlp.pipes as eds
from confit.utils.random import set_seed

set_seed(42)

nlp = edsnlp.blank("eds")
nlp.add_pipe(
    eds.ner_crf(  # (1)!
        mode="joint",  # (2)!
        target_span_getter="ml-ner",  # (3)!
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
3. The `target_span_getter` parameter defines the name of the span group used to train the NER model. We will need to make sure the entities from the training dataset are assigned to this span group (next section).
4. The word embeddings used by the CRF are computed by a CNN, which builds on top of another embedding layer.
5. The base embedding layer is a pretrained transformer, which computes contextualized word embeddings.
6. We chose the `prajjwal1/bert-tiny` model in this tutorial for testing purposes, but we recommend using a larger model like `bert-base-cased` or `camembert-base` (French) for real-world applications.

### 2. Adapting a dataset

To train a pipeline, we must convert our annotated data into documents that will be either used as training samples or a evaluation samples. This is done by applying to function to the raw data to convert entries
into a list of Doc objects. We will assume the dataset has been annotated using [Brat](https://brat.nlplab.org), but any format can be used.

At this step, we might also want to perform data augmentation, filtering, splitting or any other data transformation. Note that this function will be used to load both the training data and the test
data. Here we will split on line jumps and filter out empty documents from the training data.

```{ .python .no-check }
import edsnlp


def skip_empty_docs(batch):
    for doc in batch:
        if len(doc.ents) > 0:
            yield doc


training_data = (
    # Read the data from the brat directory and convert it into Docs,
    edsnlp.data.read_standoff(
        train_data_path,
        # Store spans in default "ents", and "ml-ner" for the training (prev. section)
        span_setter=["ents", "ml-ner"],
        # Tokenize the training docs with the same tokenizer as the trained model
        tokenizer=nlp.tokenizer,
    )
    # Split the documents on line jumps
    .map(eds.split(regex="\n\n"))
    # Filter out empty documents
    .map_batches(skip_empty_docs)
    # Add any other transformation if needed
)
```

However, we will keep all the documents in the validation data, even empty docs, to obtain representative metrics.

```{ .python .no-check }
val_data = edsnlp.data.read_standoff(
    val_data_path,
    span_setter=["ents", "ml-ner"],
    tokenizer=nlp.tokenizer,
)
val_docs = list(val_data)  # execute and convert the stream to a list
```

### 4. Complete the initialization with the training data

We initialize the missing or incomplete components attributes (such as label vocabularies) with the training dataset

```{ .python .no-check }
nlp.post_init(training_data)
```

### 5. Preprocessing the data

The training dataset is then preprocessed into features. The resulting preprocessed dataset is then wrapped into a pytorch DataLoader to be fed to the model during the training loop with the model's own collate method. We will use EDS-NLP's [Streams][edsnlp.core.stream.Stream] to handle the data processing.

Loop on the training data (same as `loop=True` in the `read_standoff` method). Note that this will
loop before shuffling or any further preprocessing step, meaning these operations will be applied every epoch. This is usually a good thing if preprocessing contains randomness to increase the diversity of
the training samples while avoiding loading multiple versions of a same document in memory. To loop after preprocessing, we can collect the stream into a list and loop on the list (`edsnlp.data.from_iterable(training_data), loop=True`).

```{ .python .no-check }
batches = training_data.loop()
```

Apply shuffling to our stream. If our dataset is too large to fit in memory, instead of "dataset" we can set the shuffle batch size to "100 docs" for example, or "fragment" for parquet datasets.

```{ .python .no-check }
batches = batches.shuffle("dataset")
```

```{ .python .no-check }
# We can now preprocess the data
batches = batches.map(
   nlp.preprocess,  # (1)!
   kwargs={"supervision": True}
)
```

1. This will call the `preprocess_supervised` method of the [TorchComponent][edsnlp.core.torch_component.TorchComponent] class and return a nested dictionary containing the required features and labels.

Make batches of at most 8192 tokens and assemble (or "collate") the samples into a batch

```{ .python .no-check }
from edsnlp.utils.batching import stat_batchify
batches = batches.batchify(batch_size=8192, batch_by=stat_batchify("tokens")  # (1)!
batches = batches.map(nlp.collate, kwargs={"device": device})
```

1. We must make sure that a feature produced by `preprocess` contains the string "tokens".

and that's it ! We now have a looping stream of batches that we can feed to our model.
For better efficiency, we can also perform the preprocessing step in parallel in a separate worker by using `num_cpu_workers` option on our stream.

```{ .python .no-check }
batches = batches.set_processing(num_cpu_workers=1, process_start_method="spawn")  # (1)!
```

1. Since we use a GPU, we must use the "spawn" method to create the workers. This is because the default multiprocessing "fork" method is not compatible with CUDA.

### 6. The training loop

We instantiate an optimizer and start the training loop

```{ .python .no-check }
from itertools import chain, repeat
from tqdm import tqdm

lr = 3e-4
max_steps = 400

# Move the model to the GPU if available (device = "cuda")
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

### 7. Optimizing the weights

Inside the training loop, the trainable components are fed the collated batches from the dataloader by calling
the [`TorchComponent.forward`][edsnlp.core.torch_component.TorchComponent.forward] method (via a simple call) to compute the losses. In the case we train a multi-task model (not in this tutorial), the
outputs of shared embedding are reused between components, we enable caching by wrapping this step in a cache context. The training loop is otherwise carried in a similar fashion to a standard pytorch
training loop

```{ .python .no-check }
    with nlp.cache():
        loss = torch.zeros((), device=device)
        for name, component in nlp.torch_components():
            output = component(batch[name])  # (1)!
            if "loss" in output:
                loss += output["loss"]

    loss.backward()

    optimizer.step()
```

### 8. Evaluating the model

Finally, the model is evaluated on the validation dataset and saved at regular intervals.

```{ .python .no-check }
from edsnlp.metrics.ner import NerExactMetric
from copy import deepcopy

metric = NerExactMetric(span_getter=nlp.pipes.ner.target_span_getter)

    ...

    if ((step + 1) % 100) == 0:
        with nlp.select_pipes(enable=["ner"]):  # (1)!
            preds = deepcopy(val_docs)

            # Clean the documents that our model will annotate
            for doc in preds:
                doc.ents = doc.spans["ml-ner"] = []
            preds = nlp.pipe(preds)  # (2)!
            print(metric(val_docs, preds))

    nlp.to_disk("model")  # (3)!
```

1. In the case we have multiple pipes in our model, we may want to selectively evaluate each pipe, thus we use the `select_pipes` method to disable every pipe except "ner".
2. We use the `pipe` method to run the "ner" component on the validation dataset. This method is similar to the `__call__` method of EDS-NLP components, but it is used to run a component on a list of
   Docs. This is also equivalent to
    ```{ .python .no-check }
    preds = (
        edsnlp.data
       .from_iterable(preds)
       .map_pipeline(nlp)
    )
    ```
3. We could also have saved the model with `torch.save(model, "model.pt")`, but `nlp.to_disk` avoids pickling and allows to inspect the model's files by saving them into a structured directory.

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
        batch_size: int = 8192,
        lr: float = 1e-4,
        max_steps: int = 400,
        num_preprocessing_workers: int = 1,
        evaluation_interval: int = 100,
    ):
        device = "cuda"

        # Define function to skip empty docs
        def skip_empty_docs(batch: Iterator) -> Iterator:
            for doc in batch:
                if len(doc.ents) > 0:
                    yield doc

        # Load and process training data
        training_data = (
            edsnlp.data.read_standoff(
                train_data_path,
                span_setter=["ents", "ml-ner"],
                tokenizer=nlp.tokenizer,
            )
            .map(eds.split(regex="\n\n"))
            .map_batches(skip_empty_docs)
        )

        # Load validation data
        val_data = edsnlp.data.read_standoff(
            val_data_path,
            span_setter=["ents", "ml-ner"],
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
                        doc.ents = doc.spans["ml-ner"] = []
                    preds = nlp.pipe(preds)
                    print(metric(val_docs, preds))

                nlp.to_disk("model")


    if __name__ == "__main__":
        nlp = edsnlp.blank("eds")
        nlp.add_pipe(
            eds.ner_crf(
                mode="joint",
                target_span_getter="ml-ner",
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
            batch_size=8192,
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
  lang: "eds"
  components:
    ner:
      "@factory": "eds.ner_crf"
      mode: "joint"
      target_span_getter: "ml-ner"
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
  batch_size: 8192
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

This tutorial gave you a glimpse of the training API of EDS-NLP. To build a custom trainable component, you can refer to the [TorchComponent][edsnlp.core.torch_component.TorchComponent] class or look up the implementation of some of the trainable components on GitHub.
