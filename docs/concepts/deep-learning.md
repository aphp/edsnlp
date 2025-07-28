# Deep Learning

Under the hood, EDS-NLP uses PyTorch to train and run deep-learning models. EDS-NLP acts as a sidekick to PyTorch, providing a set of tools to perform preprocessing, composition and evaluation. The trainable [`TorchComponents`][edsnlp.core.torch_component.TorchComponent] are actually PyTorch modules with a few extra methods to handle the feature preprocessing and postprocessing. Therefore, EDS-NLP is fully compatible with the PyTorch ecosystem.

To build and train a deep learning model, you can either build a training script from scratch (check out the [*Make a training script*](/tutorials/make-a-training-script) tutorial), or use the provided training API. The training API is designed to be flexible and can handle various types of models, including Named Entity Recognition (NER) models, span classifiers, and more. However, if you need more control over the training process, consider writing your own training script.

EDS-NLP supports training models either from the command line or from a Python script or notebook, and switching between the two is relatively straightforward thanks to the use of [Confit](https://aphp.github.io/confit/).

??? note "A word about Confit"

    EDS-NLP makes heavy use of [Confit](https://aphp.github.io/confit/), a configuration library that allows you call functions from Python or the CLI, and validate and optionally cast their arguments.

    The EDS-NLP function described on this page is the `train` function of the `edsnlp.train` module. When passing a dict to a type-hinted argument (either from a `config.yml` file, or by calling the function in Python), Confit will instantiate the correct class with the arguments provided in the dict. For instance, we pass a dict to the `train_data` parameter, which is actually type hinted as a `TrainingData`: this dict will actually be used as keyword arguments to instantiate this `TrainingData` object. You can also instantiate a `TrainingData` object directly and pass it to the function.

    You can also tell Confit specifically which class you want to instantiate by using the `@register_name = "name_of_the_registered_class"` key and value in a dict or config section. We make a heavy use of this mechanism to build pipeline architectures.

## How it works

To train a model with EDS-NLP, you need the following ingredients:

- **Pipeline**: a [pipeline][edsnlp.core.pipeline.Pipeline] with at least one trainable component. Components that share parameters or that must be updated together are trained in the same phase.

- **Training streams**: one or more streams of documents wrapped in a TrainingData object. Each of these specifies how to shuffle the stream, how to batch it with a stat expression such as `2000 words` or `16 spans`, whether to split batches into sub batches for gradient accumulation, and which components it feeds.

- **Validation streams**: optional streams of documents used for periodic evaluation.

- **Scorer**: a [scorer][edsnlp.training.trainer.GenericScorer] that defines the metrics to compute on the validation set. By default, it reports speed and uses autocast during scoring unless disabled.

- **Optimizer**: an [optimizer][edsnlp.training.optimizer.ScheduledOptimizer]. Defaults to AdamW with linear warmup and two groups of parameters, one for the transformer with lr 5•10^-5, and one for the rest of the model with lr 3•10^-4.

- **A bunch of hyperparameters**: finally, the function expects various hyperparameters (most of them set to sensible defaults) to the function, such as `max_steps`, `seed`, `validation_interval`, `checkpoint_interval`, `grad_max_norm`, and more.

The training then proceeds in several steps:

**Setup**
The function prepares the device with [Accelerate](https://huggingface.co/docs/accelerate/index), creates the output folders, materializes the validation set from the user-provided stream, and runs a post-initialization pass on the training data when requested. This `post_init` op let's the pipeline inspect the data before learning to adjust the number of heads depending on the labels encountered. Finally, the optimizer is instantiated.

**Phases**
Training runs **by phases**. A phase groups components that should be optimized together because they share parameters (think for instance of a BERT shared between multiple models). During a phase, losses are computed for each of these "active" components at each step, and only their parameters are updated.

**Data preparation**
Each TrainingData object turns its streams of documents into device ready batches. It optionally shuffles the stream, preprocess the documents for the active components, builds stat-aware batches (for instance, limiting the number of tokens per batch), optionally splits batches into sub batches for gradient accumulation, then converts everything into device-ready tensors. This can be done in parallel to the actual deep-learning work.

**Optimization**
For every training step the function draws one batch from each training stream (in case there are more than one) and synchronizes statistics across processes (in case we're doing multi-GPU training) to keep supports and losses consistent. It runs forward passes for the phase components. When several components reuse the same intermediate features a cache avoids recomputation. Gradients are accumulated over sub batches.

**Gradient safety**
Gradients are always clipped to `grad_max_norm`. Optionally the function tracks an exponential moving mean and variance of the gradient norm. If a spike is detected you can clip to the running mean or to a threshold or skip the update depending on `grad_dev_policy`. This protects training from rare extreme updates.

**Validation and logging**
At regular intervals the scorer evaluates the pipeline on the validation documents. It isolates each task by copying docs and disabling unrelated pipes to avoid leakage. It reports throughput and metrics for NER and span attribute classifiers plus any custom metrics.

**Checkpoints and output**
The model is saved on schedule and at the end in `output_dir/model-last` unless saving is disabled.

## Tutorials and examples

--8<-- "docs/tutorials/index.md:deep-learning-tutorials"

## Parameters of `edsnlp.train` {: #edsnlp.training.trainer.train }

Here are the parameters you can pass to the `train` function:

::: edsnlp.training.trainer.train
    options:
        heading_level: 4
        only_parameters: no-header
        skip_parameters: []
        show_source: false
        show_toc: false
