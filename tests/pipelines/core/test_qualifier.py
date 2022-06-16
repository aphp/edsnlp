from pathlib import Path

import spacy
from pytest import fixture
from spacy.cli.train import train as spacy_train
from spacy.tokens import DocBin

from edsnlp.utils.examples import parse_example

CONFIG = """
[paths]
train = null
dev = null
raw = null
init_tok2vec = null
vectors = null

[system]
seed = 42
gpu_allocator = null

[nlp]
lang = "eds"
pipeline = ["tok2vec","qualifier"]
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {"@tokenizers":"spacy.Tokenizer.v1"}
batch_size = 100

[components]

[components.qualifier]
factory = "qualifier"
threshold = 0.5

[components.qualifier.model]
@architectures = "edsnlp.qualifier_model.v1"

[components.qualifier.model.classification_layer]
@architectures = "edsnlp.classification_layer.v1"
nI = null
nO = null

[components.qualifier.model.create_tensor]
@architectures = "edsnlp.qualifier_tensor.v1"
pooling = {"@layers":"reduce_mean.v1"}

[components.qualifier.model.create_tensor.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${components.tok2vec.model.width}
upstream = "*"

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.HashEmbedCNN.v1"
pretrained_vectors = null
width = 96
depth = 2
embed_size = 2000
window_size = 1
maxout_pieces = 3
subword_features = true

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.train]
@readers = "edsnlp.ents_corpus.v1"
file = ${paths.train}

[training]
seed = ${system.seed}
gpu_allocator = ${system.gpu_allocator}
dropout = 0.1
accumulate_gradient = 1
patience = 2
max_epochs = 0
max_steps = 100
eval_frequency = 5
frozen_components = []
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
before_to_disk = null
annotating_components = []

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001
learn_rate = 0.001

[training.score_weights]
qual_micro_p = 0.0
qual_micro_r = 0.0
qual_micro_f = 1.0

[pretraining]

[initialize]
vectors = ${paths.vectors}
init_tok2vec = ${paths.init_tok2vec}
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
"""


entities = ["malade", "diabétique", "fatigué", "triste", "en état de choque"]
base_examples = [
    "Le patient n'est pas <ent negation=True>{entity}</ent>",
    "Le patient est <ent negation=False>{entity}</ent>",
]

examples = []
for entity in entities:
    for example in base_examples:
        examples.append(example.format(entity=entity))


@fixture
def docbin():

    nlp = spacy.blank("eds")
    db = DocBin(store_user_data=True)

    for example in examples:
        text, entities = parse_example(example)
        doc = nlp(text)

        ents = []

        for entity in entities:
            ent = doc.char_span(entity.start_char, entity.end_char, label="ent")
            ent._.qualifiers = {m.key: m.value for m in entity.modifiers}
            ents.append(ent)

        doc.ents = ents

        db.add(doc)

    return db


@fixture
def train(tmp_path: Path, docbin: DocBin):
    path = tmp_path / "train.spacy"
    docbin.to_disk(path)
    return path


@fixture
def dev(tmp_path: Path, docbin: DocBin):
    path = tmp_path / "dev.spacy"
    docbin.to_disk(path)
    return path


@fixture
def config(tmp_path: Path):
    path = tmp_path / "config.cfg"
    path.write_text(CONFIG)
    return path


def test_qualifier_training(config, train, dev):
    spacy_train(
        str(config),
        overrides={"paths.train": str(train), "paths.dev": str(dev)},
    )
