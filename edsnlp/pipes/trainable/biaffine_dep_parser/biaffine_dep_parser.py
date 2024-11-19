from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast

import foldedtensor as ft
import numpy as np
import torch
import torch.nn.functional as F
from spacy.tokens import Doc, Span
from typing_extensions import Literal

from edsnlp.core import PipelineProtocol
from edsnlp.core.torch_component import BatchInput, BatchOutput, TorchComponent
from edsnlp.pipes.trainable.embeddings.typing import WordEmbeddingComponent
from edsnlp.utils.span_getters import SpanGetterArg, get_spans

logger = logging.getLogger(__name__)


# ===============================================================
def chuliu_edmonds_one_root(scores: np.ndarray) -> np.ndarray:
    """
    Shamelessly copied from
    https://github.com/hopsparser/hopsparser/blob/main/hopsparser/mst.py#L63
    All credits, Loic Grobol at Université Paris Nanterre, France, the
    original author of this implementation. Find the license of the hopsparser software
    below:

    Copyright 2020 Benoît Crabbé benoit.crabbe@linguist.univ-paris-diderot.fr

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in the
    Software without restriction, including without limitation the rights to use,
    copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
    Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
    PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
    OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    ---

    Repeatedly Use the Chu‑Liu/Edmonds algorithm to find a maximum spanning
    dependency tree from the weight matrix of a rooted weighted directed graph.

    **ATTENTION: this modifies `scores` in place.**

    ## Input

    - `scores`: A 2d numeric array such that `scores[i][j]` is the weight
    of the `$j→i$` edge in the graph and the 0-th node is the root.

    ## Output

    - `tree`: A 1d integer array such that `tree[i]` is the head of the `i`-th node
    """

    # FIXME: we don't actually need this in CLE: we only need one critical cycle
    def tarjan(tree: np.ndarray) -> List[np.ndarray]:
        """Use Tarjan's SCC algorithm to find cycles in a tree

        ## Input

        - `tree`: A 1d integer array such that `tree[i]` is the head of
           the `i`-th node

        ## Output

        - `cycles`: A list of 1d bool arrays such that `cycles[i][j]` is
           true iff the `j`-th node of
          `tree` is in the `i`-th cycle
        """
        indices = -np.ones_like(tree)
        lowlinks = -np.ones_like(tree)
        onstack = np.zeros_like(tree, dtype=bool)
        stack = list()
        # I think this is in a list to be able to mutate it in the closure, even
        # though `nonlocal` exists
        _index = [0]
        cycles = []

        def strong_connect(i):
            _index[0] += 1
            index = _index[-1]  # `_index` is of length 1 so this is also `_index[0]`???
            indices[i] = lowlinks[i] = index - 1
            stack.append(i)
            onstack[i] = True
            dependents = np.where(np.equal(tree, i))[0]
            for j in dependents:
                if indices[j] == -1:
                    strong_connect(j)
                    lowlinks[i] = min(lowlinks[i], lowlinks[j])
                elif onstack[j]:
                    lowlinks[i] = min(lowlinks[i], indices[j])

            # There's a cycle!
            if lowlinks[i] == indices[i]:
                cycle = np.zeros_like(indices, dtype=bool)
                while stack[-1] != i:
                    j = stack.pop()
                    onstack[j] = False
                    cycle[j] = True
                stack.pop()
                onstack[i] = False
                cycle[i] = True
                if cycle.sum() > 1:
                    cycles.append(cycle)
            return

        # -------------------------------------------------------------
        for i in range(len(tree)):
            if indices[i] == -1:
                strong_connect(i)
        return cycles

    # TODO: split out a `contraction` function to make this more readable
    def chuliu_edmonds(scores: np.ndarray) -> np.ndarray:
        """Use the Chu‑Liu/Edmonds algorithm to find a maximum spanning
        arborescence from the weight matrix of a rooted weighted directed
        graph

        ## Input

        - `scores`: A 2d numeric array such that `scores[i][j]` is the
          weight of the `$j→i$` edge in the graph and the 0-th node is the root.

        ## Output

        - `tree`: A 1d integer array such that `tree[i]` is the head of the `i`-th node
        """
        np.fill_diagonal(scores, -float("inf"))  # prevent self-loops
        scores[0] = -float("inf")
        scores[0, 0] = 0
        tree = cast(np.ndarray, np.argmax(scores, axis=1))
        cycles = tarjan(tree)
        if not cycles:
            return tree
        else:
            # t = len(tree); c = len(cycle); n = len(noncycle)
            # locations of cycle; (t) in [0,1]
            cycle = cycles.pop()
            # indices of cycle in original tree; (c) in t
            cycle_locs = np.where(cycle)[0]
            # heads of cycle in original tree; (c) in t
            cycle_subtree = tree[cycle]
            # scores of cycle in original tree; (c) in R
            cycle_scores = scores[cycle, cycle_subtree]
            # total score of cycle; () in R
            total_cycle_score = cycle_scores.sum()

            # locations of noncycle; (t) in [0,1]
            noncycle = np.logical_not(cycle)
            # indices of noncycle in original tree; (n) in t
            noncycle_locs = np.where(noncycle)[0]

            # scores of cycle's potential heads; (c x n) - (c) + () -> (n x c) in R
            metanode_head_scores = (
                scores[cycle][:, noncycle]
                - cycle_scores[:, np.newaxis]
                + total_cycle_score
            )
            # scores of cycle's potential dependents; (n x c) in R
            metanode_dep_scores = scores[noncycle][:, cycle]
            # best noncycle head for each cycle dependent; (n) in c
            metanode_heads = np.argmax(metanode_head_scores, axis=0)
            # best cycle head for each noncycle dependent; (n) in c
            metanode_deps = np.argmax(metanode_dep_scores, axis=1)

            # scores of noncycle graph; (n x n) in R
            subscores = scores[noncycle][:, noncycle]
            # expand to make space for the metanode (n+1 x n+1) in R
            subscores = np.pad(subscores, ((0, 1), (0, 1)), "constant")
            # set the contracted graph scores of cycle's potential
            # heads; (c x n)[:, (n) in n] in R -> (n) in R
            subscores[-1, :-1] = metanode_head_scores[
                metanode_heads, np.arange(len(noncycle_locs))
            ]
            # set the contracted graph scores of cycle's potential
            # dependents; (n x c)[(n) in n] in R-> (n) in R
            subscores[:-1, -1] = metanode_dep_scores[
                np.arange(len(noncycle_locs)), metanode_deps
            ]

            # MST with contraction; (n+1) in n+1
            contracted_tree = chuliu_edmonds(subscores)
            # head of the cycle; () in n
            cycle_head = contracted_tree[-1]
            # fixed tree: (n) in n+1
            contracted_tree = contracted_tree[:-1]
            # initialize new tree; (t) in 0
            new_tree = -np.ones_like(tree)
            # fixed tree with no heads coming from the cycle: (n) in [0,1]
            contracted_subtree = contracted_tree < len(contracted_tree)
            # add the nodes to the new tree (t)[(n)[(n) in [0,1]] in t]
            # in t = (n)[(n)[(n) in [0,1]] in n] in t
            new_tree[noncycle_locs[contracted_subtree]] = noncycle_locs[
                contracted_tree[contracted_subtree]
            ]
            # fixed tree with heads coming from the cycle: (n) in [0,1]
            contracted_subtree = np.logical_not(contracted_subtree)
            # add the nodes to the tree (t)[(n)[(n) in [0,1]] in t]
            # in t = (c)[(n)[(n) in [0,1]] in c] in t
            new_tree[noncycle_locs[contracted_subtree]] = cycle_locs[
                metanode_deps[contracted_subtree]
            ]
            # add the old cycle to the tree; (t)[(c) in t] in t = (t)[(c) in t] in t
            new_tree[cycle_locs] = tree[cycle_locs]
            # root of the cycle; (n)[() in n] in c = () in c
            cycle_root = metanode_heads[cycle_head]
            # add the root of the cycle to the new
            # tree; (t)[(c)[() in c] in t] = (c)[() in c]
            new_tree[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]
            return new_tree

    scores = scores.astype(np.float64)
    tree = chuliu_edmonds(scores)
    roots_to_try = np.where(np.equal(tree[1:], 0))[0] + 1

    # PW: small change here (<= instead of ==) to avoid crashes in pathological cases
    if len(roots_to_try) <= 1:
        return tree

    # -------------------------------------------------------------
    def set_root(scores: np.ndarray, root: int) -> Tuple[np.ndarray, np.ndarray]:
        """Force the `root`-th node to be the only node under the root by overwriting
        the weights of the other children of the root."""
        root_score = scores[root, 0]
        scores = np.array(scores)
        scores[1:, 0] = -float("inf")
        scores[root] = -float("inf")
        scores[root, 0] = 0
        return scores, root_score

    # We find the maximum spanning dependency tree by trying every possible root
    best_score, best_tree = -np.inf, None  # This is what's causing it to crash
    for root in roots_to_try:
        _scores, root_score = set_root(scores, root)
        _tree = chuliu_edmonds(_scores)
        tree_probs = _scores[np.arange(len(_scores)), _tree]
        tree_score = (
            (tree_probs).sum() + (root_score)
            if (tree_probs > -np.inf).all()
            else -np.inf
        )
        if tree_score > best_score:
            best_score = tree_score
            best_tree = _tree

    assert best_tree is not None
    return best_tree


class MLP(torch.nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float = 0.0
    ):
        super().__init__()
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class BiAffine(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features,
    ):
        super().__init__()
        self.bilinear = torch.nn.Bilinear(
            in_features, in_features, out_features, bias=True
        )
        self.head_linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.tail_linear = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, u, v):
        scores = torch.einsum("bux,bvy,lxy->buvl", u, v, self.bilinear.weight)
        scores = scores + self.head_linear(u).unsqueeze(2)
        scores = scores + self.tail_linear(v).unsqueeze(1)
        scores = scores + self.bilinear.bias
        return scores


class TrainableBiaffineDependencyParser(
    TorchComponent[BatchOutput, BatchInput],
):
    """
    The `eds.biaffine_dep_parser` component is a trainable dependency parser
    based on a biaffine model ([@dozat2017deepbiaffineattentionneural]). For each
    token, the model predicts a score for each possible head in the document, and a
    score for each possible label for each head. The results are then decoded either
    greedily by picking the best scoring head for each token independently, or
    holistically by computing the Maximum Spanning Tree (MST) over the graph of
    token → head scores.

    !!! warning "Experimental"

        This component is experimental. In particular, it expects the input to be
        sentences and not full documents, as it has not been optimized for memory
        efficiency yet and computed the full matrix of scores for all pairs of tokens
        in a document.

        At the moment, it is mostly used for benchmarking and research purposes.

    Examples
    --------
    ```{ .python }
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.biaffine_dep_parser(
            embedding=eds.transformer(model="prajjwal1/bert-tiny"),
            hidden_size=128,
            dropout_p=0.1,
            # labels unset, will be inferred from the data in `post_init`
            decoding_mode="mst",
        ),
        name="dep_parser"
    )
    ```

    Dependency parsers are typically trained on CoNLL-formatted
    [Universal Dependencies corpora](https://universaldependencies.org/#download),
    which you can load using the [`edsnlp.data.read_conll`][edsnlp.data.read_conll]
    function.

    To train the model, refer to the [Training tutorial](/tutorials/training).

    Parameters
    ----------
    nlp: Optional[PipelineProtocol]
        The pipeline object
    name: str
        Name of the component
    embedding: WordEmbeddingComponent
        The word embedding component
    context_getter: Optional[SpanGetterArg]
        What context to use when computing the span embeddings (defaults to the whole
        document). For example `{"section": "conclusion"}` to predict dependencies
        in the conclusion section of documents.
    use_attrs: Optional[List[str]]
        The attributes to use as features for the model (ex. `["pos_"]` to use the POS
        tag). By default, no attributes are used.

        Note that if you train a model with attributes, you will need to provide the
        same attributes during inference, and the model might not work well if the
        attributes were not annotated accurately on the test data.
    attr_size: int
        The size of the attribute embeddings.
    hidden_size: int
        The size of the hidden layer in the MLP.
    dropout_p: float
        The dropout probability to use in the MLP.
    labels: List[str]
        The labels to predict. The labels can also be inferred from the data during
        `nlp.post_init(...)`.
    decoding_mode: Literal["greedy", "mst"]
        Whether to decode the dependencies greedily or using the Maximum Spanning Tree
        algorithm.

    Authors and citation
    --------------------
    The `eds.biaffine_dep_parser` trainable pipe was developed by
    AP-HP's Data Science team, and heavily inspired by the implementation of
    [@grobol:hal-03223424]. The biaffine architecture is based on the biaffine parser
    of [@dozat2017deepbiaffineattentionneural].
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol] = None,
        name: str = "biaffine_dep_parser",
        *,
        embedding: WordEmbeddingComponent,
        context_getter: Optional[SpanGetterArg] = None,
        use_attrs: Optional[List[str]] = None,
        attr_size: int = 32,
        hidden_size: int = 128,
        dropout_p: float = 0.0,
        labels: List[str] = ["root"],
        decoding_mode: Literal["greedy", "mst"] = "mst",
    ):
        super().__init__(nlp=nlp, name=name)
        self.embedding = embedding
        self.use_attrs: List[str] = use_attrs or []
        self.labels = list(labels) or []
        self.labels_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.context_getter = context_getter
        cat_dim = self.embedding.output_size + len(self.use_attrs) * attr_size
        self.head_mlp = MLP(cat_dim, hidden_size, hidden_size, dropout_p)
        self.tail_mlp = MLP(cat_dim, hidden_size, hidden_size, dropout_p)
        self.arc_biaffine = BiAffine(hidden_size, 1)
        self.lab_biaffine = BiAffine(hidden_size, len(self.labels))
        self.root_embed = torch.nn.Parameter(torch.randn(cat_dim)[None, None, :])
        self.decoding_mode = decoding_mode
        self.attr_vocabs = {attr: [] for attr in self.use_attrs}
        self.attr_to_idx = {attr: {} for attr in self.use_attrs}
        self.attr_embeddings = torch.nn.ModuleDict(
            {
                attr: torch.nn.Embedding(len(vocab), attr_size)
                for attr, vocab in self.attr_vocabs.items()
            }
        )

    def update_labels(self, labels: Sequence[str], attrs: Dict[str, List[str]]):
        old_labs = self.labels if self.labels is not None else ()
        old_index = torch.as_tensor(
            [i for i, lab in enumerate(old_labs) if lab in labels], dtype=torch.long
        )
        new_index = torch.as_tensor(
            [labels.index(lab) for lab in old_labs if lab in labels], dtype=torch.long
        )
        new_biaffine = BiAffine(self.arc_biaffine.bilinear.in1_features, len(labels))
        # fmt: off
        new_biaffine.bilinear.weight.data[new_index] = self.arc_biaffine.bilinear.weight.data[old_index]  # noqa: E501
        new_biaffine.bilinear.bias.data[new_index] = self.arc_biaffine.bilinear.bias.data[old_index]  # noqa: E501
        new_biaffine.head_linear.weight.data[new_index] = self.arc_biaffine.head_linear.weight.data[old_index]  # noqa: E501
        new_biaffine.tail_linear.weight.data[new_index] = self.arc_biaffine.tail_linear.weight.data[old_index]  # noqa: E501
        # fmt: on
        self.lab_biaffine.bilinear.weight.data = new_biaffine.bilinear.weight.data
        self.lab_biaffine.bilinear.bias.data = new_biaffine.bilinear.bias.data
        self.lab_biaffine.head_linear.weight.data = new_biaffine.head_linear.weight.data
        self.lab_biaffine.tail_linear.weight.data = new_biaffine.tail_linear.weight.data
        self.labels = labels
        self.labels_to_idx = {lab: i for i, lab in enumerate(labels)}

        for attr, vals in attrs.items():
            emb = self.attr_embeddings[attr]
            old_vals = (
                self.attr_vocabs[attr] if self.attr_vocabs[attr] is not None else ()
            )
            old_index = torch.as_tensor(
                [i for i, val in enumerate(old_vals) if val in vals], dtype=torch.long
            )
            new_index = torch.as_tensor(
                [vals.index(val) for val in old_vals if val in vals], dtype=torch.long
            )
            new_emb = torch.nn.Embedding(
                len(vals), self.attr_embeddings[attr].weight.size(1)
            )
            new_emb.weight.data[new_index] = emb.weight.data[old_index]
            self.attr_embeddings[attr].weight.data = new_emb.weight.data
            self.attr_vocabs[attr] = vals
            self.attr_to_idx[attr] = {val: i for i, val in enumerate(vals)}

    def post_init(self, gold_data: Iterable[Doc], exclude: Set[str]):
        super().post_init(gold_data, exclude=exclude)
        labels = dict()
        attr_vocabs = {attr: dict() for attr in self.use_attrs}
        for doc in gold_data:
            ctxs = (
                get_spans(doc, self.context_getter) if self.context_getter else [doc[:]]
            )
            for ctx in ctxs:
                for token in ctx:
                    labels[token.dep_] = True
                    for attr in self.use_attrs:
                        attr_vocabs[attr][getattr(token, attr)] = True
        self.update_labels(
            labels=list(labels.keys()),
            attrs={attr: list(v.keys()) for attr, v in attr_vocabs.items()},
        )

    def preprocess(self, doc: Doc, **kwargs) -> Dict[str, Any]:
        ctxs = get_spans(doc, self.context_getter) if self.context_getter else [doc[:]]
        prep = {
            "embedding": self.embedding.preprocess(doc, contexts=ctxs, **kwargs),
            "$contexts": ctxs,
            "stats": {
                "dep_words": 0,
            },
        }
        for attr in self.use_attrs:
            prep[attr] = [
                [self.attr_to_idx[attr].get(getattr(token, attr), 0) for token in ctx]
                for ctx in ctxs
            ]
        for ctx in ctxs:
            prep["stats"]["dep_words"] += len(ctx) + 1
        return prep

    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        preps = self.preprocess(doc)
        arc_targets = []  # head idx for each token
        lab_targets = []  # arc label idx for each token
        for ctx in preps["$contexts"]:
            ctx_start = ctx.start
            arc_targets.append(
                [
                    0,
                    *(
                        token.head.i - ctx_start + 1 if token.head.i != token.i else 0
                        for token in ctx
                    ),
                ]
            )
            lab_targets.append([0, *(self.labels_to_idx[t.dep_] for t in ctx)])
        return {
            **preps,
            "arc_targets": arc_targets,
            "lab_targets": lab_targets,
        }

    def collate(self, preps: Dict[str, Any]) -> BatchInput:
        collated = {"embedding": self.embedding.collate(preps["embedding"])}
        collated["stats"] = {
            k: sum(v) for k, v in preps["stats"].items() if not k.startswith("__")
        }
        for attr in self.use_attrs:
            collated[attr] = ft.as_folded_tensor(
                preps[attr],
                data_dims=("context", "tail"),
                full_names=("sample", "context", "tail"),
                dtype=torch.long,
            )
        if "arc_targets" in preps:
            collated["arc_targets"] = ft.as_folded_tensor(
                preps["arc_targets"],
                data_dims=("context", "tail"),
                full_names=("sample", "context", "tail"),
                dtype=torch.long,
            )
        if "lab_targets" in preps:
            collated["lab_targets"] = ft.as_folded_tensor(
                preps["lab_targets"],
                data_dims=("context", "tail"),
                full_names=("sample", "context", "tail"),
                dtype=torch.long,
            )

        return collated

    # noinspection SpellCheckingInspection
    def forward(self, batch: BatchInput) -> BatchOutput:
        embeds = self.embedding(batch["embedding"])["embeddings"]
        embeds = embeds.refold("context", "word")
        embeds = torch.cat(
            [
                embeds,
                *(self.attr_embeddings[attr](batch[attr]) for attr in self.use_attrs),
            ],
            dim=-1,
        )

        embeds_with_root = torch.cat(
            [
                self.root_embed.expand(embeds.shape[0], 1, self.root_embed.size(-1)),
                embeds,
            ],
            dim=1,
        )

        tail_embeds = self.tail_mlp(embeds_with_root)  # (contexts, tail=words, dim)
        head_embeds = self.head_mlp(embeds_with_root)  # (contexts, words+root, dim)

        # Scores: (contexts, tail=1+words, head=1+words)
        arc_logits = self.arc_biaffine(tail_embeds, head_embeds).squeeze(-1)
        # Scores: (contexts, tail=1+words, head=1+words, labels)
        lab_logits = self.lab_biaffine(tail_embeds, head_embeds)

        if "arc_targets" in batch:
            num_labels = lab_logits.shape[-1]
            arc_targets = batch["arc_targets"]  # (contexts, tail=1+words) -> head_idx
            lab_targets = batch["lab_targets"]
            # arc_targets: (contexts, tail=1+words) -> head_idx
            flat_arc_logits = (
                # (contexts, tail=1+words, head=1+words)
                arc_logits.masked_fill(~arc_targets.mask[:, None, :], -10000)[
                    # -> (all_flattened_tails_with_root, head=1+words)
                    arc_targets.mask
                ]
            )
            flat_arc_targets = arc_targets[arc_targets.mask]
            arc_loss = (
                F.cross_entropy(
                    # (all_flattened_tails_with_root, head_with_root)
                    flat_arc_logits,
                    flat_arc_targets,
                    reduction="sum",
                )
                / batch["stats"]["dep_words"]
            )
            flat_lab_logits = (
                # lab_logits: (contexts, tail=1+words, head=1+words, labels)
                lab_logits[arc_targets.mask]
                # -> (all_flattened_tails_with_root, head=1+words, labels)
                .gather(1, flat_arc_targets[:, None, None].expand(-1, 1, num_labels))
                # -> (all_flattened_tails_with_root, 1, labels)
                .squeeze(1)
                # -> (all_flattened_tails_with_root, labels)
            )
            # TODO, directly in collate
            flat_lab_targets = (
                lab_targets[
                    # (contexts, tail=1+words) -> label
                    arc_targets.mask
                ]
                # (all_flattened_tails_with_root) -> label
            )
            lab_loss = (
                F.cross_entropy(
                    flat_lab_logits,
                    flat_lab_targets,
                    reduction="sum",
                )
                / batch["stats"]["dep_words"]
            )
            return {
                "arc_loss": arc_loss,
                "lab_loss": lab_loss,
                "loss": arc_loss + lab_loss,
            }
        else:
            return {
                "arc_logits": arc_logits,
                "arc_labels": lab_logits.argmax(-1),
            }

    def postprocess(
        self,
        docs: Sequence[Doc],
        results: BatchOutput,
        inputs: List[Dict[str, Any]],
    ) -> Sequence[Doc]:
        # Preprocessed docs should still be in the cache
        # (context, head=words + 1, tail=words + 1), ie head -> tail
        contexts = [ctx for sample in inputs for ctx in sample["$contexts"]]

        for ctx, arc_logits, arc_labels in zip(
            contexts,
            results["arc_logits"].detach().cpu().numpy(),
            results["arc_labels"].detach().cpu(),
        ):
            ctx: Span
            if self.decoding_mode == "greedy":
                tail_to_head_idx = arc_logits.argmax(-1)
            else:
                tail_to_head_idx = chuliu_edmonds_one_root(arc_logits)
                tail_to_head_idx = torch.as_tensor(tail_to_head_idx)

            # lab_logits: (tail=words+1, head=words+1, labels) -> prob
            # arc_logits: (tail=words+1) -> head_idx
            labels = arc_labels[torch.arange(arc_labels.shape[0]), tail_to_head_idx]

            # Set arc and dep rel on the Span context
            for tail_idx, (head_idx, label) in enumerate(
                zip(tail_to_head_idx.tolist(), labels.tolist())
            ):
                if head_idx == 0:
                    continue
                head = ctx[head_idx - 1]
                tail = ctx[tail_idx - 1]
                tail.head = head
                tail.dep_ = self.labels[label]

        return docs
