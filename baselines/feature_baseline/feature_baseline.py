# feature_prediction.py
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Dict, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.impute import SimpleImputer


# -----------------------------
# Load annotations
# -----------------------------
def load_annotations(ann_folder: str) -> Dict[str, pd.DataFrame]:
    """
    Load learner annotations.
    Each file: word<TAB>label
    Returns dict: learner_id -> DataFrame[word, label]
    learner_id is the file name without extension.
    """
    learners = {}
    for fname in os.listdir(ann_folder):
        fpath = os.path.join(ann_folder, fname)
        if not os.path.isfile(fpath):
            continue
        learner_id = os.path.splitext(fname)[0]  # e.g. "0", "1"
        words, labels = [], []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                words.append(parts[0].lower())
                labels.append(int(parts[1]))
        learners[learner_id] = pd.DataFrame({"word": words, "label": labels})
    return learners

# -----------------------------
# Classifier factory
# -----------------------------
def get_model(name: str):
    if name == "lr":
        return LogisticRegression(max_iter=200, class_weight="balanced")
    if name == "svm":
        return SVC(probability=True, class_weight="balanced")
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=10)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced")
    raise ValueError(f"Unknown model: {name}")

# -----------------------------
# Learner feature computation
# -----------------------------
def compute_learner_features(train_df: pd.DataFrame, bands_meta: Dict) -> Dict[str, float]:
    """
    Compute learner-level features from training annotations.
    train_df must contain: label, band_id
    bands_meta must contain: phi (dict band_id -> weight)
    """
    # simple known rate
    known_rate = train_df["label"].mean()

    # phi-weighted ability
    phi = bands_meta.get("phi", {})
    phi_score = 0.0
    for band_id, group in train_df.groupby("band_id"):
        prop_known = group["label"].mean()
        phi_score += phi.get(int(band_id), 0.0) * prop_known

    return {"learner_known_rate": known_rate, "learner_phi": phi_score}

# -----------------------------
# Run training & evaluation for one learner
# -----------------------------
from sklearn.impute import SimpleImputer

def run_for_learner(learner_id: str,
                    ann_df: pd.DataFrame,
                    feat_df: pd.DataFrame,
                    model_name: str,
                    label_num: int,
                    bands_meta: Dict,
                    random_state: int = 42):

    # join features with labels
    df = ann_df.merge(feat_df, on="word", how="inner")
    if len(df) < label_num + 1:
        return None  # not enough data

    # split train/test
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    train_idx = idx[:label_num]
    test_idx = idx[label_num:]

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # compute learner-level features from training labels only
    learner_feats = compute_learner_features(train_df, bands_meta)
    for k, v in learner_feats.items():
        train_df[k] = v
        test_df[k] = v

    # separate labels
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # drop non-feature columns
    drop_cols = ["word", "label"]
    X_train_df = train_df.drop(columns=drop_cols)
    X_test_df = test_df.drop(columns=drop_cols)

    # numeric preprocessing: replace infs -> NaN, then impute missing, then scale
    numeric_train = X_train_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    numeric_test = X_test_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="mean")
    numeric_train = imputer.fit_transform(numeric_train)
    numeric_test = imputer.transform(numeric_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(numeric_train)
    X_test = scaler.transform(numeric_test)

    # train + predict
    model = get_model(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics = {
        "user_id": learner_id,
        "acc": acc,
        "f1_macro": f1,
        "mcc": mcc,
    }

    classification_result = {
        w: {"true": int(t), "pred": int(p)}
        for w, t, p in zip(test_df["word"], y_test, y_pred)
    }

    return metrics, classification_result

# -----------------------------
# Main experiment runner
# -----------------------------
def run_prediction(annotations_folder: str,
                   feature_file: str,
                   bands_meta_file: str,
                   out_csv: str,
                   out_json: str,
                   model_name: str = "lr",
                   label_num: int = 50,
                   random_state: int = 42):
    feat_df = pd.read_parquet(feature_file)
    with open(bands_meta_file, "r", encoding="utf-8") as f:
        bands_meta = json.load(f)
    learners = load_annotations(annotations_folder)

    all_results = []
    all_preds = {}

    for learner_id, ann_df in tqdm(learners.items(), desc="Learners"):
        res = run_for_learner(learner_id, ann_df, feat_df, model_name, label_num, bands_meta, random_state)
        if res is None:
            raise RuntimeError("problem with {learner_id}")
            continue
        metrics, preds = res
        all_results.append(metrics)
        all_preds[learner_id] = preds

    # aggregate row
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        avg = results_df[["acc", "f1_macro", "mcc"]].mean().to_dict()
        avg["user_id"] = "average"
        results_df = pd.concat([results_df, pd.DataFrame([avg])], ignore_index=True)

    # save
    results_df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_preds, f, indent=2)

    print(f"Saved metrics to {out_csv}")
    print(f"Saved predictions to {out_json}")

    return results_df


if __name__ == "__main__":
    for model in ["lr", "svm", "knn", "rf"]:
        print(model)
        results = run_prediction(
            annotations_folder="./evkd",        # your annotations (0.txt, 1.txt...)
            feature_file="./evkd_features/word_features_evkd.parquet",     # from feature extractor
            bands_meta_file="./evkd_features/bands_meta_evkd.json",        # from feature extractor
            out_csv=f"./evkd_results/{model}.csv",
            out_json=f"./evkd_results/{model}_predictions.json",
            model_name=model,   # lr, svm, knn, rf
            label_num=50,
        )
    # print(results)
