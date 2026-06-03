# word_feature_pipeline.py
import os
import re
import math
import json
from collections import Counter, defaultdict
from typing import Dict, Iterable, Set, Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans

# --- syllables (CMU -> heuristic fallback)
import nltk
nltk.data.path.append('./nltk_data')
# nltk.download("cmudict", download_dir='./nltk_data')
CMU = nltk.corpus.cmudict.dict()

WORD_RE = re.compile(r"[a-z]+")

# -----------------------------
# 0) Utilities
# -----------------------------
def count_syllables(word: str) -> int:
    w = word.lower()
    if w in CMU:
        # min phones among variants
        return min(len([ph for ph in pron if ph[-1].isdigit()]) for pron in CMU[w])
    # heuristic: count vowel groups; ensure at least 1
    s = len(re.findall(r"[aeiouy]+", w))
    return s if s > 0 else 1

def orthographic_feats(word: str) -> Dict[str, float]:
    return {
        "len_chars": float(len(word)),
        "len_syllables": float(count_syllables(word)),
    }

# -----------------------------
# 1) Gather vocabulary from annotation files
# -----------------------------
def gather_vocab_from_annotations(ann_root: str) -> Set[str]:
    """
    Each file contains lines: word<TAB>label
    Return the unique set of words (lowercased).
    """
    vocab = set()
    for fname in tqdm(os.listdir(ann_root), desc="Scanning annotation files"):
        fpath = os.path.join(ann_root, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if not parts:
                    continue
                w = parts[0].strip().lower()
                if WORD_RE.fullmatch(w):  # keep alphabetic words only
                    vocab.add(w)
    return vocab

# -----------------------------
# 2) COCA processing (line = document)
# -----------------------------
def coca_counts_for_vocab(coca_root: str, target_vocab: Set[str]) -> Tuple[Counter, Counter, int]:
    """
    Iterate COCA: genre folders -> files -> lines (documents).
    Only count tokens that are in target_vocab.
    Returns: (token_freq, doc_freq, total_tokens_seen)
    """
    token_freq = Counter()
    doc_freq = Counter()
    total_tokens = 0

    genre_dirs = [d for d in os.listdir(coca_root) if os.path.isdir(os.path.join(coca_root, d))]
    for genre in tqdm(genre_dirs, desc="COCA genres"):
        gpath = os.path.join(coca_root, genre)
        files = [f for f in os.listdir(gpath) if os.path.isfile(os.path.join(gpath, f))]
        for fname in tqdm(files, desc=f"{genre} files", leave=False):
            fpath = os.path.join(gpath, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    # one document per line
                    text = line.strip().lower()
                    if not text or not re.search(r"[a-z]", text):
                        continue
                    toks = WORD_RE.findall(text)
                    if not toks:
                        continue
                    # filter to target vocab
                    toks = [t for t in toks if t in target_vocab]
                    if not toks:
                        continue
                    total_tokens += len(toks)
                    counts = Counter(toks)
                    token_freq.update(counts)
                    # doc freq: each word counts once per line
                    for w in counts.keys():
                        doc_freq[w] += 1

    return token_freq, doc_freq, total_tokens

def build_coca_df(token_freq: Counter, doc_freq: Counter, total_tokens: int) -> pd.DataFrame:
    """
    Assemble COCA frequency dataframe for the words we saw.
    """
    rows = []
    for w, f in token_freq.items():
        df = doc_freq.get(w, 0)
        rows.append({
            "word": w,
            "coca_freq": float(f),
            "coca_log_freq": float(math.log1p(f)),
            "coca_doc_freq": float(df),
            # Zipf-ish scale: log10(per-billion); protect zeros with max(1,total_tokens)
            "coca_zipf": float(math.log10((f / max(1, total_tokens)) * 1e9)) if f > 0 and total_tokens > 0 else -np.inf,
        })
    return pd.DataFrame(rows)

# -----------------------------
# 3) MRC loader (useful columns)
# -----------------------------
def load_mrc(path: str) -> pd.DataFrame:
    """
    Load the MRC psycholinguistic database and keep informative columns.
    """
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    rename = {
        "Word": "word",
        "Number of Letters": "mrc_num_letters",
        "Number of Phonemes": "mrc_num_phonemes",
        "Number of Syllables": "mrc_num_syllables",
        "KF Written Frequency": "mrc_kf_freq",
        "KF Number of Categories": "mrc_kf_categories",
        "KF Number of Samples": "mrc_kf_samples",
        "Thorndike-Lorge Frequency": "mrc_tl_freq",
        "Brown Verbal Frequency": "mrc_brown_freq",
        "Familiarity": "mrc_familiarity",
        "Concreteness": "mrc_concreteness",
        "Imageability": "mrc_imageability",
        "Meaningfulness: Coloradao Norms": "mrc_meaningfulness_colorado",
        "Meaningfulness: Pavio Norms": "mrc_meaningfulness_pavio",
        "Age of Acquisition Rating": "mrc_aoa",
        "Word Type": "mrc_word_type",
        "Comprehensive Syntactic Category": "mrc_syntactic_cat",
        "Common Part of Speech": "mrc_pos",
        "Morphemic status": "mrc_morphemic_status",
        "Contextual Status": "mrc_contextual_status",
        "Pronunciation Variability": "mrc_pron_variability",
        "Capitalization": "mrc_capitalization",
        "Irregular Plural": "mrc_irregular_plural",
    }
    # keep only present columns
    keep_cols = [c for c in rename.keys() if c in df.columns]
    df = df[keep_cols].rename(columns={k: rename[k] for k in keep_cols})
    df["word"] = df["word"].str.lower()
    # Deduplicate by aggregating per word
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.difference(["word"])

    agg_dict = {c: "mean" for c in num_cols}
    for c in cat_cols:
        agg_dict[c] = lambda x: x.mode().iloc[0] if not x.mode().empty else "UNK"

    df = df.groupby("word", as_index=False).agg(agg_dict)

    # index by word
    return df.set_index("word")

# -----------------------------
# 4) CEFR loader (optional)
# -----------------------------
def load_cefr_map(path: Optional[str]) -> Dict[str, str]:
    """
    Load CEFR mapping from a plain txt file: word<TAB>level
    """
    if not path:
        raise RuntimeError("no path provided")
    cdf = pd.read_csv(path, delimiter="\t", header=None, names=["word", "cefr_level"])
    cdf["word"] = cdf["word"].astype(str).str.strip().str.lower()
    return dict(zip(cdf["word"], cdf["cefr_level"]))

# -----------------------------
# 5) Frequency bands (+ phi weights meta)
# -----------------------------
def add_frequency_bands(coca_df: pd.DataFrame, k: int = 5, method: str = "kmeans") -> Tuple[pd.DataFrame, Dict]:
    """
    Add band_id for each word using coca_log_freq, and compute band meta:
      - band_mean_freq
      - s_max (mean of top-10 freqs)
      - phi weights (inverse-freq style as in Avdiu et al.)
    """
    df = coca_df.copy()
    x = df["coca_log_freq"].values.reshape(-1, 1)

    if method == "kmeans":
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        band_id = km.fit_predict(x)
        df["band_id"] = band_id.astype(np.int16)
        # band means based on raw freq (not log)
        band_mean = df.groupby("band_id")["coca_freq"].mean().to_dict()
    else:  # quantiles
        q = pd.qcut(df["coca_log_freq"], q=k, labels=False, duplicates="drop")
        df["band_id"] = q.astype(np.int16)
        band_mean = df.groupby("band_id")["coca_freq"].mean().to_dict()

    # s_max: mean of top-10 frequencies (within this vocab)
    top10 = df["coca_freq"].nlargest(min(10, len(df))).mean() if len(df) else 0.0
    # phi weights
    numer = {b: (top10 - band_mean[b]) for b in band_mean}
    denom = sum(max(0.0, v) for v in numer.values()) or 1.0
    phi = {b: max(0.0, numer[b]) / denom for b in band_mean}

    # also store phi on each row (handy later; constant within a band)
    df["band_phi"] = df["band_id"].map(phi).astype(np.float32)

    meta = {
        "k": k,
        "method": method,
        "band_mean_freq": {int(k): float(v) for k, v in band_mean.items()},
        "s_max_top10_mean": float(top10),
        "phi": {int(k): float(v) for k, v in phi.items()},
    }
    return df, meta

# -----------------------------
# 6) Main: build word -> feature table
# -----------------------------
def build_word_feature_table(
    annotations_root: str,
    coca_root: str,
    mrc_path: str,
    out_parquet: str = "word_features.parquet",
    out_meta_json: str = "bands_meta.json",
    cefr_path: Optional[str] = None,
    k_bands: int = 10,
    band_method: str = "kmeans",
) -> pd.DataFrame:
    # 6.1 vocab
    print("loading vocab...")
    vocab = gather_vocab_from_annotations(annotations_root)
    if not vocab:
        raise RuntimeError("No words found in annotations.")
    # 6.2 COCA counts restricted to vocab
    cache_path = "coca_freq.parquet"
    if os.path.exists(cache_path):
        print(f"Loading COCA frequencies from {cache_path}")
        coca_df = pd.read_parquet(cache_path)
        print("loaded coca frequency")
    else:

        tok_freq, doc_freq, total_tokens = coca_counts_for_vocab(coca_root, vocab)
        coca_df = build_coca_df(tok_freq, doc_freq, total_tokens)
        coca_df.to_parquet(cache_path, index=False)
        print(f"Saved COCA frequencies to {cache_path}")
 
    # ensure we have all vocab words represented (fill zeros for unseen)
    missing = sorted(vocab - set(coca_df["word"]))
    if missing:
        coca_df = pd.concat([coca_df, pd.DataFrame({
            "word": missing,
            "coca_freq": 0.0,
            "coca_log_freq": 0.0,
            "coca_doc_freq": 0.0,
            "coca_zipf": -np.inf,
        })], ignore_index=True)

    # 6.3 add bands + phi meta
    coca_df, meta = add_frequency_bands(coca_df, k=k_bands, method=band_method)

    # 6.4 MRC + CEFR
    mrc_df = load_mrc(mrc_path)           # index=word
    cefr_map = load_cefr_map(cefr_path)   # dict word->level
    
 
    # 6.5 assemble feature rows
    rows = []
    for _, row in tqdm(coca_df.iterrows(), total=len(coca_df), desc="Extracting per-word features"):
        w = row["word"]
        feats = {
            "word": w,
            # COCA numeric
            "coca_freq": row["coca_freq"],
            "coca_log_freq": row["coca_log_freq"],
            "coca_doc_freq": row["coca_doc_freq"],
            "coca_zipf": row["coca_zipf"],
            "band_id": int(row["band_id"]),
            "band_phi": float(row["band_phi"]),
        }
        # orthography (lightweight)
        feats.update(orthographic_feats(w))

        # MRC merge (if present)
        if w in mrc_df.index:
            m = mrc_df.loc[w]
            # numeric columns (cast to float32 later)
            mrc_numeric = [
                "mrc_num_letters","mrc_num_phonemes","mrc_num_syllables",
                "mrc_kf_freq","mrc_kf_categories","mrc_kf_samples",
                "mrc_tl_freq","mrc_brown_freq","mrc_familiarity",
                "mrc_concreteness","mrc_imageability",
                "mrc_meaningfulness_colorado","mrc_meaningfulness_pavio","mrc_aoa",
            ]
            for c in mrc_numeric:
                if c in mrc_df.columns:
                    feats[c] = float(m[c]) if pd.notna(m[c]) else np.nan
            # categorical
            mrc_cat = [
                "mrc_word_type","mrc_syntactic_cat","mrc_pos",
                "mrc_morphemic_status","mrc_contextual_status",
                "mrc_pron_variability","mrc_capitalization","mrc_irregular_plural",
            ]
            for c in mrc_cat:
                if c in mrc_df.columns:
                    feats[c] = m[c] if pd.notna(m[c]) else "UNK"
        else:
            # fill missing MRC entries
            for c in [
                "mrc_num_letters","mrc_num_phonemes","mrc_num_syllables",
                "mrc_kf_freq","mrc_kf_categories","mrc_kf_samples",
                "mrc_tl_freq","mrc_brown_freq","mrc_familiarity",
                "mrc_concreteness","mrc_imageability",
                "mrc_meaningfulness_colorado","mrc_meaningfulness_pavio","mrc_aoa",
            ]:
                feats[c] = np.nan
            for c in [
                "mrc_word_type","mrc_syntactic_cat","mrc_pos",
                "mrc_morphemic_status","mrc_contextual_status",
                "mrc_pron_variability","mrc_capitalization","mrc_irregular_plural",
            ]:
                feats[c] = "UNK"

        # CEFR (optional)
        feats["cefr_level"] = cefr_map.get(w, "UNK")

        rows.append(feats)

    df = pd.DataFrame(rows)

    # 6.6 enforce dtypes & save
    numeric_cols = [
        "coca_freq","coca_log_freq","coca_doc_freq","coca_zipf","band_phi",
        "len_chars","len_syllables",
        "mrc_num_letters","mrc_num_phonemes","mrc_num_syllables",
        "mrc_kf_freq","mrc_kf_categories","mrc_kf_samples",
        "mrc_tl_freq","mrc_brown_freq","mrc_familiarity",
        "mrc_concreteness","mrc_imageability",
        "mrc_meaningfulness_colorado","mrc_meaningfulness_pavio","mrc_aoa",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # small ints / categories
    df["band_id"] = df["band_id"].astype("int16")
    for c in [
        "mrc_word_type","mrc_syntactic_cat","mrc_pos",
        "mrc_morphemic_status","mrc_contextual_status",
        "mrc_pron_variability","mrc_capitalization","mrc_irregular_plural",
        "cefr_level",
    ]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    df["word"] = df["word"].astype("string")

    df.to_parquet(out_parquet, index=False)
    with open(out_meta_json, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    print(f"✅ Saved {len(df):,} words × {df.shape[1]-1} features -> {out_parquet}")
    print(f"📄 Band meta (phi, s_max, means) -> {out_meta_json}")
    return df

# -----------------------------
# Example CLI-ish usage
# -----------------------------
if __name__ == "__main__":

    # FEATURES = build_word_feature_table(
    #     annotations_root="./evkd",   # files with word<TAB>label
    #     coca_root="path/to/COCA_text_root",             # genre folders -> text files -> lines
    #     mrc_path="./mrc.csv",           
    #     out_parquet="./evkd_features/word_features_evkd.parquet",
    #     out_meta_json="./evkd_features/bands_meta_evkd.json",
    #     cefr_path="./cefr_words.txt",                        # optionally point to your EVP mapping CSV
    #     k_bands=10,
    #     band_method="kmeans",                  # or "quantile"
    # )
    df = pd.read_parquet("./evkd_features/word_features_evkd.parquet")
    print(df.head())
    print(df.columns.values)