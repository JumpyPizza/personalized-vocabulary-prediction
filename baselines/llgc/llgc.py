"""
Ehara et al. (EMNLP 2014),

Inputs
------
--labels_dir     : folder with user files; each line:  word<TAB>0/1  (1=known, 0=not)
                   All files share the SAME word list and order.
--bnc_freq       : TSV word<TAB>rank (1=most frequent). Missing ranks handled (see below).
--coca_freq      : TSV word<TAB>rank (optional; required if graph_type=bnc_coca)
--graph_type     : 'bnc' or 'bnc_coca'
--input_samples  : number of seed words to “ask” per experiment (e.g., 10–50)
--mu             : LLGC regularization (default 0.01)
--bins           : number of frequency bins (default 10)
--out_dir        : where to write predictions and metrics

Note that we implement based on the paper's description; 
we cannot guarantee that it remains 100% faithful/correct to the original paper.
"""

import argparse
import os
import csv
import json
import numpy as np
from typing import List, Dict, Tuple

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report

from scipy.sparse import csr_matrix, coo_matrix, diags, identity
from scipy.sparse.linalg import spsolve, eigsh


# ---------------------------
# I/O
# ---------------------------
def read_labels(labels_dir: str) -> Tuple[List[str], List[str], np.ndarray]:
    files = sorted([f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))])
    if not files:
        raise FileNotFoundError("No files found in labels_dir")
    users = [os.path.splitext(f)[0] for f in files]

    vocab = []
    with open(os.path.join(labels_dir, files[0]), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip(): continue
            w, _ = line.rstrip("\n").split("\t")
            vocab.append(w.strip().lower())

    n_words = len(vocab)
    L = np.zeros((n_words, len(users)), dtype=np.int8)
    for u_idx, f in enumerate(files):
        with open(os.path.join(labels_dir, f), encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip(): continue
                _, lab = line.rstrip("\n").split("\t")
                L[i, u_idx] = 1 if lab.strip() == "1" else -1
    return vocab, users, L


def read_freq_ranks(path: str) -> Dict[str, int]:
    ranks = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            w, r = line.rstrip("\n").split("\t")[:2]
            try:
                ranks[w.strip().lower()] = int(r)
            except:
                pass
    return ranks


# ---------------------------
# Bin ALL words by frequency rank
# ---------------------------
def order_indices_by_rank(vocab: List[str], ranks: Dict[str, int]) -> np.ndarray:
    """
    Return an index array that sorts vocab by ascending rank.
    Words with missing rank get a very large rank but preserve original order.
    """
    n = len(vocab)
    large = 10**12
    rank_arr = np.empty(n, dtype=np.int64)
    for i, w in enumerate(vocab):
        rank_arr[i] = ranks.get(w, large)
    # stable argsort preserves input order among ties / missing ranks
    return np.argsort(rank_arr, kind="mergesort")


def split_into_bins(order_idx: np.ndarray, n_bins: int) -> List[np.ndarray]:
    """
    Split 'order_idx' (permutation of [0..n-1]) into n_bins contiguous, as-even-as-possible bins.
    EVERY index is assigned to exactly one bin.
    """
    n = len(order_idx)
    base = n // n_bins
    rem = n % n_bins
    bins = []
    start = 0
    for b in range(n_bins):
        size = base + (1 if b < rem else 0)
        bins.append(order_idx[start:start+size])
        start += size
    return bins


def build_multi_complete_from_bins(n: int, bin_idx_lists: List[np.ndarray]) -> csr_matrix:
    """
    Build a sparse adjacency with edges=1 within each bin (complete subgraphs).
    """
    rows, cols = [], []
    for idxs in bin_idx_lists:
        if idxs.size == 0: continue
        arr = idxs.astype(np.int32)
        m = arr.size
        rr = np.repeat(arr, m)
        cc = np.tile(arr, m)
        mask = rr != cc
        rows.append(rr[mask])
        cols.append(cc[mask])
    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.ones(rows.shape[0], dtype=np.float32)
        W = coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    else:
        W = csr_matrix((n, n), dtype=np.float32)
    # symmetrize & zero diag (should already be)
    W = (W + W.T) * 0.5
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def build_graphs(vocab: List[str], bnc_ranks: Dict[str, int], coca_ranks: Dict[str, int] = None,
                 n_bins: int = 8) -> Tuple[csr_matrix, List[np.ndarray], csr_matrix]:
    """
    Build BNC (and optional COCA) multi-complete graphs with ALL words binned.
    Returns: (W_bnc, bnc_bins, W_coca or None)
    """
    order_bnc = order_indices_by_rank(vocab, bnc_ranks)
    bnc_bins = split_into_bins(order_bnc, n_bins)
    W_bnc = build_multi_complete_from_bins(len(vocab), bnc_bins)

    W_coca = None
    if coca_ranks is not None:
        order_coca = order_indices_by_rank(vocab, coca_ranks)
        coca_bins = split_into_bins(order_coca, n_bins)
        W_coca = build_multi_complete_from_bins(len(vocab), coca_bins)

    return W_bnc, bnc_bins, W_coca


def merge_graphs_sparse(graphs: List[csr_matrix]) -> csr_matrix:
    W = None
    for G in graphs:
        if G is None: continue
        W = G if W is None else (W + G)
    if W is None:
        raise ValueError("No graphs to merge.")
    W = (W + W.T) * 0.5
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


# ---------------------------
# LLGC (normalized Laplacian solve)
# ---------------------------
def normalized_laplacian_sparse(W: csr_matrix) -> csr_matrix:
    n = W.shape[0]
    d = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
    d[d == 0] = 1.0
    Dmhalf = diags(1.0 / np.sqrt(d))
    S = Dmhalf @ W @ Dmhalf
    I = identity(n, format="csr", dtype=np.float64)
    return I - S


def llgc_solve(W: csr_matrix, y: np.ndarray, mu: float = 0.01, ridge: float = 1e-8) -> np.ndarray:
    """
    Solve (I + mu L_norm) f = y   with sparse spsolve.
    y should be in {+1, -1, 0}.
    """
    n = W.shape[0]
    Lnorm = normalized_laplacian_sparse(W)
    A = identity(n, format="csr", dtype=np.float64) * (1.0 + ridge) + mu * Lnorm
    f = spsolve(A, y.astype(np.float64))
    return f


# ---------------------------
# Sampling
# ---------------------------
def sample_round_robin(bin_idx_lists: List[np.ndarray], k: int, rng=None) -> List[int]:
    """
    Round-robin across bins; within a chosen bin, pick uniformly from unused.
    Guarantees no duplicates; uses all words across bins if k>sum(bin sizes) (then returns all).
    """
    rng = rng or np.random.default_rng()
    B = len(bin_idx_lists)
    counts = np.zeros(B, dtype=np.int32)
    selected = []
    used = set()

    total = sum(len(b) for b in bin_idx_lists)
    k = min(k, total)

    while len(selected) < k:
        minc = counts.min()
        cand_bins = np.flatnonzero(counts == minc)
        b = int(rng.choice(cand_bins))
        pool = [int(i) for i in bin_idx_lists[b] if int(i) not in used]
        if not pool:
            counts[b] += 1  # mark as “exhausted” for balancing and continue
            continue
        i = int(rng.choice(pool))
        selected.append(i)
        used.add(i)
        counts[b] += 1
    return selected


def sample_gu_han(W: csr_matrix, k: int, mu: float = 0.01, eig_k: int = None,
                  eps: float = 1e-6, rng=None) -> List[int]:
    """
    Gu & Han (2012) non-interactive graph-based active sampling (k picks).
    Runs in a k_eig-dimensional space using smallest eigenpairs of L_norm.
    """
    rng = rng or np.random.default_rng()
    n = W.shape[0]
    if eig_k is None:
        eig_k = int(min(128, max(2, n - 1)))
    eig_k = max(2, min(eig_k, n - 1))

    Lnorm = normalized_laplacian_sparse(W)
    lam, U = eigsh(Lnorm, k=eig_k, which="SM", tol=1e-3)
    order = np.argsort(lam)
    lam = lam[order]
    U = U[:, order]
    lam = np.where(lam < eps, eps, lam)
    lam_inv = 1.0 / lam
    h = (mu * lam + 1.0) ** 2 - 1.0
    Hinv = np.diag(h)
    UT = U.T

    selected = np.zeros(n, dtype=bool)
    chosen = []

    k = min(k, n)
    for _ in range(k):
        V = Hinv @ UT                       # (k_eig × n)
        # scores = num / den, vectorized:
        tmp = (np.sqrt(lam_inv)[:, None] * V)
        num = np.sum(tmp * tmp, axis=0)     # (n,)
        den = 1.0 + np.sum(U * V.T, axis=1) # (n,)
        scores = num / den
        scores[selected] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            break
        chosen.append(best)
        selected[best] = True
        # rank-1 update
        u = U[best, :].reshape(-1, 1)
        Hu = Hinv @ u
        denom = float(1.0 + (u.T @ Hu))
        Hinv = Hinv - (Hu @ Hu.T) / denom
    return chosen


# ---------------------------
# Runner
# ---------------------------
def run(labels_dir: str,
        bnc_freq: str,
        coca_freq: str,
        graph_type: str,
        input_samples: int,
        mu: float = 0.01,
        bins: int = 8,
        out_dir: str = None,
        eig_k: int = None):

    vocab, users, L01 = read_labels(labels_dir)
    n = len(vocab)
    # Convert 0/1 to -1/+1 labels for LLGC targets
    L = np.where(L01 == 1, 1, -1).astype(np.int8)

    # Build graphs (ALL words binned)
    bnc_ranks = read_freq_ranks(bnc_freq)
    W_bnc, bnc_bins, W_coca = build_graphs(
        vocab, bnc_ranks, read_freq_ranks(coca_freq) if (graph_type == "bnc_coca" and coca_freq) else None, n_bins=bins
    )

    if graph_type == "bnc":
        W = W_bnc
        seeds = sample_round_robin(bnc_bins, input_samples)
    elif graph_type == "bnc_coca":
        if W_coca is None:
            raise ValueError("--coca_freq is required for graph_type=bnc_coca")
        W = merge_graphs_sparse([W_bnc, W_coca])
        seeds = sample_gu_han(W, input_samples, mu=mu, eig_k=eig_k)
    else:
        raise ValueError("graph_type must be 'bnc' or 'bnc_coca'")

    seeds = np.array(seeds, dtype=int)
    seed_mask = np.zeros(n, dtype=bool); seed_mask[seeds] = True
    nonseed_mask = ~seed_mask

    out_dir = out_dir or os.path.join(labels_dir, "predictions_allwords")
    os.makedirs(out_dir, exist_ok=True)

    # Precompute LLGC matrix pieces once
    # (We solve per-user because y differs per user.)
    metrics_csv = []
    report_json = {}
    for u_idx, user in enumerate(users):
        y = L[:, u_idx].astype(np.float64)
        y_seeded = np.zeros_like(y)
        y_seeded[seed_mask] = y[seed_mask]   # only seeds are labeled; others unlabeled (=0)

        f = llgc_solve(W, y_seeded, mu=mu)
        y_pred = (f > 0).astype(int)         # strict: 1 iff f>0
        gold = (L[:, u_idx] == 1).astype(int)

        # Evaluate on ALL non-training words (ALL non-seeds)
        g_eval = gold[nonseed_mask]
        p_eval = y_pred[nonseed_mask]

        acc = accuracy_score(g_eval, p_eval)
        f1m = f1_score(g_eval, p_eval, average="macro", zero_division=0)
        mcc = matthews_corrcoef(g_eval, p_eval)
        metrics_csv.append([user, acc, f1m, mcc])
        report_json[user] = classification_report(
            g_eval, p_eval, output_dict=True, zero_division=0
        )
        print(f"[{user}] acc={acc:.3f}  f1_macro={f1m:.3f}  mcc={mcc:.3f}")

        # # Write per-word predictions (for transparency)
        # with open(os.path.join(out_dir, f"{user}_pred.tsv"), "w", encoding="utf-8") as fo:
        #     fo.write("word\tseed\tgold\tpred\tscore\n")
        #     for i, w in enumerate(vocab):
        #         fo.write(f"{w}\t{int(seed_mask[i])}\t{gold[i]}\t{y_pred[i]}\t{f[i]:.6f}\n")
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f_csv:
        import csv
        writer = csv.writer(f_csv)
        writer.writerow(["user_id", "acc", "f1_macro", "mcc"])
        writer.writerows(metrics_csv)

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f_json:
        import json
        json.dump(report_json, f_json, indent=2)
    # # Save seeds and config
    # with open(os.path.join(out_dir, "seeds.json"), "w", encoding="utf-8") as fs:
    #     json.dump({
    #         "graph_type": graph_type,
    #         "input_samples": int(input_samples),
    #         "mu": float(mu),
    #         "bins": int(bins),
    #         "seeds": seeds.tolist()
    #     }, fs, indent=2)


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--bnc_freq", required=True)
    ap.add_argument("--coca_freq", default=None)
    ap.add_argument("--graph_type", choices=["bnc", "bnc_coca"], required=True)
    ap.add_argument("--input_samples", type=int, required=True)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--eig_k", type=int, default=None, help="k eigenpairs for Gu&Han (bnc_coca)")
    args = ap.parse_args()

    run(
        labels_dir=args.labels_dir,
        bnc_freq=args.bnc_freq,
        coca_freq=args.coca_freq,
        graph_type=args.graph_type,
        input_samples=args.input_samples,
        mu=args.mu,
        bins=args.bins,
        out_dir=args.out_dir,
        eig_k=args.eig_k,
    )
