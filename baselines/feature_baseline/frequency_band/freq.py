
import os
import numpy as np
import pandas as pd
from wordfreq import zipf_frequency
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score
from tqdm import tqdm
import json
# # note that how to binarize labels have relatively large impact on the performance:
# # only 5 is known vs 4, 5 is known
# also, the samples from the band have non-neglectable impact (seed 0 vs 42)

# -----------------------------
# Utilities
# -----------------------------
def compute_frequency_buckets(words, n_buckets=10):
    freqs = [zipf_frequency(w, "en") if isinstance(w, str) else 0.0 for w in words]
    thresholds = [np.percentile(freqs, i) for i in range(10, 100, int(100 / n_buckets))]

    def assign_bucket(freq):
        for idx, th in enumerate(thresholds):
            if freq < th:
                return idx
        return len(thresholds)

    buckets = [assign_bucket(f) for f in freqs]
    return freqs, buckets

def majority_vote(arr):
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]

def sample_bucket_indices(df, n=5, random_state=42):
    """
    Return dict: bucket -> np.array of sampled row indices (not labels).
    Uses up to n rows per bucket.
    """
    sampled = {}
    rng = np.random.default_rng(random_state)
    for bucket, group in df.groupby("bucket"):
        if len(group) == 0:
            continue
        k = min(n, len(group))
        sampled[bucket] = rng.choice(group.index.to_numpy(), size=k, replace=False)
    return sampled

# -----------------------------
# Main Evaluation
# -----------------------------
def evaluate_band_method(data_folder, user_group_map, n=5, n_buckets=10, exclude_train_from_eval=True, random_state=0):
    # 1) Load all annotators
    user_dfs = {}
    all_words = set()
    for fname in os.listdir(data_folder):
        if not fname.endswith(".txt"):
            continue
        uid = os.path.splitext(fname)[0]
        df = pd.read_csv(os.path.join(data_folder, fname), sep="\t", names=["word", "label"])
        df["word"] = df["word"].astype(str).str.lower()
        user_dfs[uid] = df
        all_words.update(df["word"].tolist())

    all_words = sorted(all_words)

    # 2) Frequency buckets (global)
    freqs, buckets = compute_frequency_buckets(all_words, n_buckets=n_buckets)
    w2f = dict(zip(all_words, freqs))
    w2b = dict(zip(all_words, buckets))

    # 3) Attach freq/bucket
    for uid, df in user_dfs.items():
        df["freq"] = df["word"].map(w2f)
        df["bucket"] = df["word"].map(w2b)

    # 4) Per-user evaluation with proper train/test split
    per_user = {}
    for uid, df in tqdm(user_dfs.items(), desc="Users"):
        df = df.reset_index(drop=True)

        # ---- training: sample indices per bucket
        sampled_idx_by_bucket = sample_bucket_indices(df, n=n, random_state=random_state)

        # bucket label = majority of sampled labels in that bucket
        bucket_label = {}
        for bkt, idxs in sampled_idx_by_bucket.items():
            lbls = df.loc[idxs, "label"].to_numpy()
            bucket_label[bkt] = majority_vote(lbls) if len(lbls) > 0 else 0

        # predictions for ALL rows (we'll filter for test below)
        preds_all = []
        for _, row in df.iterrows():
            b = row["bucket"]
            if b in bucket_label:
                preds_all.append(bucket_label[b])
            else:
                preds_all.append(0)  # default unknown if no training for that bucket
        preds_all = np.array(preds_all, dtype=int)
        true_all = df["label"].to_numpy(dtype=int)

        # ---- evaluation split
        if exclude_train_from_eval:
            train_idx = np.concatenate(list(sampled_idx_by_bucket.values())) if sampled_idx_by_bucket else np.array([], dtype=int)
            test_mask = np.ones(len(df), dtype=bool)
            test_mask[train_idx] = False
            if not np.any(test_mask):
                # fallback: if no test rows (very small user), evaluate on all
                test_mask = np.ones(len(df), dtype=bool)
        else:
            test_mask = np.ones(len(df), dtype=bool)

        y_true = true_all[test_mask]
        y_pred = preds_all[test_mask]

        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        mcc  = matthews_corrcoef(y_true, y_pred)
        rec0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

        per_user[uid] = {"accuracy": acc, "macro_f1": f1, "mcc": mcc, "recall_0": rec0}

    # 5) Aggregate
    overall = {k: float(np.mean([m[k] for m in per_user.values()])) for k in ["accuracy", "macro_f1", "mcc", "recall_0"]}

    groups = {}
    for uid, m in per_user.items():
        grp = user_group_map.get(int(uid), "unknown")
        groups.setdefault(grp, {k: [] for k in m})
        for k, v in m.items():
            groups[grp][k].append(v)
    for grp, d in groups.items():
        groups[grp] = {k: float(np.mean(v)) for k, v in d.items()}

    return {"overall": overall, "groups": groups, }


if __name__ == "__main__":
   # evkd
    # with open("../evkd_results/evkd_user_map.json", "r") as f:
    #     user_map = json.load(f)
    # user_map = {int(id):map for id, map in user_map.items()}
    # results = evaluate_band_method(
    #     data_folder="../evkd", 
    #     user_group_map=user_map, 
    #     n=5, 
    #     n_buckets=10
    # )
    # print(results)
   
    #vkd 
    user_map = {6: 'Advanced', 7: 'Advanced', 5: 'Advanced', 4: 'Beginner', 0: 'Beginner', 1: 'Beginner', 3: 'Beginner', 2: 'Advanced', 9: 'Intermediate', 8: 'Intermediate', 15: 'Beginner', 14: 'Advanced', 13: 'Beginner', 12: 'Beginner', 10: 'Intermediate', 11: 'Beginner'}
    results = evaluate_band_method(
        data_folder="../vkd_binary", 
        user_group_map=user_map, 
        n=5, 
        n_buckets=10
    )
    print(results)
