# @inproceedings{DBLP:conf/iclr/SenerS18,
#   author       = {Ozan Sener and
#                   Silvio Savarese},
#   title        = {Active Learning for Convolutional Neural Networks: {A} Core-Set Approach},
#   booktitle    = {6th International Conference on Learning Representations, {ICLR} 2018,
#                   Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings},
#   publisher    = {OpenReview.net},
#   year         = {2018},
#   url          = {https://openreview.net/forum?id=H1aIuk-RW},
# }

# diversity 

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import ActiveLearningStrategy, StrategyConfig, register_strategy


@register_strategy("coreset")
class CoreSetKCenter(ActiveLearningStrategy):
    """
    Diversity sampling via K-Center (a.k.a. Core-Set).
    We embed each sequence (mean over valid token embeddings), then
    aggregate per word (mean). Greedy k-center selection on word embeddings
    w.r.t. current labeled word embeddings.

    Note: if no labeled words exist yet, we seed the first selection by
    choosing the point with the largest 2-norm (or just the first index).
    """

    def select_words(self, model, dataset, select_k: int) -> List[str]:
        device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")

        # 1) Embeddings for UNLABELED (current dataset is the unlabeled/test pool)
        ex_embs, ex_words = sequence_embeddings(model, dataset, device, batch_size=self.cfg.batch_size)
        word2vec = {}
        for v, w in zip(ex_embs, ex_words):
            word2vec.setdefault(w, []).append(v)
        U_words = list(word2vec.keys())
        U = np.stack([np.mean(word2vec[w], axis=0) for w in U_words], axis=0)  # [Nu, D]

        # 2) seeding
        L = getattr(dataset, "labeled_bank", None)  # expected np.ndarray [Nl, D] or None
        centers = []
        if L is not None and len(L) > 0:
            # initialize min distances to nearest labeled point
            min_dist = _pairwise_distances(U, L).min(axis=1)  # [Nu]
        else:
            # seed from farthest from origin # initial
            norms = np.linalg.norm(U, axis=1)
            first = int(np.argmax(norms))
            centers.append(first)
            min_dist = _pairwise_distances(U, U[first:first+1])[:, 0]

        # 3) Greedy k-center: pick select_k new centers
        while len(centers) < select_k:
            nxt = int(np.argmax(min_dist))
            centers.append(nxt)
            d = _pairwise_distances(U, U[nxt:nxt+1])[:, 0]
            min_dist = np.minimum(min_dist, d)

        return [U_words[i] for i in centers]


def sequence_embeddings(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 128
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Return (embeddings, words) lists where each embedding is the mean over valid tokens
    (labels != -100 & attention_mask==1 & not special) from last_hidden_state.
    """
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_emb_collate,
    )

    model.eval()
    # ensure model returns hidden states (last_hidden_state always available)
    embs: List[np.ndarray] = []
    words: List[str] = []

    with torch.no_grad():
        for batch in dl:
            meta = batch.pop("meta_word")  # list[str]
            labels = batch["labels"]
            attn   = batch["attention_mask"]
            spec   = batch["special_tokens_mask"]
            batch  = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=False)
            H = outputs.last_hidden_state  # [B, T, D]
            B, T, D = H.shape

            labels = labels.cpu().numpy()
            attn   = attn.cpu().numpy()
            spec   = spec.cpu().numpy()
            Hn     = H.cpu().numpy()

            mask = (labels != -100) & (attn == 1) & (spec == 0)   # [B, T]
            for b in range(B):
                m = mask[b].astype(bool)
                if not np.any(m):
                    raise RuntimeError("no valid tokens")
                else:
                    vec = Hn[b, m].mean(axis=0).astype(np.float32)
                embs.append(vec)
                words.append(meta[b])

    return embs, words


def _emb_collate(examples):
    """
    Collate for embedding extraction (similar to scoring collate).
    """
    keys = ["input_ids", "attention_mask", "labels", "special_tokens_mask"]
    batch = {k: torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(ex[k]) for ex in examples],
        batch_first=True,
        padding_value=0 if k != "labels" else -100
    ) for k in keys}
    batch["meta_word"] = [ex["meta_word"] for ex in examples]
    return batch


def _kcenter_greedy(X: np.ndarray, k: int) -> List[int]:
    """
    Greedy K-Center on rows of X (shape [N, D]).
    Returns indices of selected centers.
    If k >= N, returns all indices.
    """
    N = X.shape[0]
    if k >= N:
        return list(range(N))
    # distances to current set (initialize with infinity)
    min_d = np.full((N,), np.inf, dtype=np.float32)

    # seed: farthest from origin (largest L2 norm)
    norms = np.linalg.norm(X, axis=1)
    first = int(np.argmax(norms))
    centers = [first]

    # update min distances
    d = _pairwise_distances(X, X[first:first+1])[:, 0]
    min_d = np.minimum(min_d, d)

    while len(centers) < k:
        # pick point with max min-distance to current centers
        nxt = int(np.argmax(min_d))
        centers.append(nxt)
        d = _pairwise_distances(X, X[nxt:nxt+1])[:, 0]
        min_d = np.minimum(min_d, d)

    return centers


def _pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Euclidean distances between rows of A [Na, D] and B [Nb, D] -> [Na, Nb]
    """
    # (A - B)^2 = A^2 + B^2 - 2AB
    A2 = np.sum(A * A, axis=1, keepdims=True)      # [Na, 1]
    B2 = np.sum(B * B, axis=1, keepdims=True).T    # [1, Nb]
    AB = A @ B.T
    D2 = np.maximum(A2 + B2 - 2.0 * AB, 0.0)
    return np.sqrt(D2, dtype=np.float32)
