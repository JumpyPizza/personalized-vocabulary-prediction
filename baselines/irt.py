# organized_cat.py
# note that this script expects input in 1-5 scales

import os
import re
import math
import json
import warnings
warnings.filterwarnings("ignore", message="This selector needs an item matrix")

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm 

import random
import numpy as np
import pandas as pd
from scipy.stats import zscore

# IRT / CAT
from cmdstanpy import CmdStanModel
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, f1_score, classification_report, accuracy_score

from catsim.selection import MaxInfoSelector, UrrySelector
from catsim.stopping import MaxItemStopper
from catsim.simulation import Estimator, Selector
from catsim import cat

# Embeddings & small MLP regressor (only for unseen words)
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torchtext.vocab import Vectors
except Exception:
    Vectors = None  # allow running without torchtext



def configure_system_threads():
    n = os.cpu_count() or 1
    os.environ.setdefault("STAN_NUM_THREADS", str(n))
    # OpenMP envs commonly read by linear algebra libs:
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(var, str(n))

def stan_cpp_options(use_opencl: bool = True, use_threads: bool = True) -> Dict[str, bool]:
    opts = {}
    if use_threads:
        opts["STAN_THREADS"] = True
    if use_opencl:
        # Stan will use OpenCL if the model & math prims support it; safe to enable.
        opts["STAN_OPENCL"] = True
        # You can target device via environment variables if desired:
        # os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        # os.environ.setdefault("OPENCL_DEVICE_ID", "0")
        # os.environ.setdefault("OPENCL_PLATFORM_ID", "0")
    return opts

# ----------------------------
# Utils
# ----------------------------

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def icc_2pl(theta: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return logistic(a * (theta - b))

def word_freq_zscores(words: List[str]) -> Dict[str, float]:
    from wordfreq import word_frequency
    raw = np.array([word_frequency(w, "en") for w in words], dtype=float)
    eps = max(np.min(raw[raw > 0]) * 0.1, 1e-12) if np.any(raw > 0) else 1e-12
    logf = np.log(np.where(raw > 0, raw, eps))
    z = zscore(logf) if np.std(logf) > 0 else np.zeros_like(logf)
    return {w: float(z[i]) for i, w in enumerate(words)}

def five_strata(words: List[str], z: Dict[str, float]) -> List[np.ndarray]:
    order = sorted(words, key=lambda w: z.get(w, 0.0))
    return np.array_split(np.array(order, dtype=object), 5)

# ----------------------------
# Data loading
# ----------------------------

def load_train_folder(folder: str | Path, n_files: int = 20) -> pd.DataFrame:
    """Load training data (UTF-8).
    Sample exactly `n_files` files for each tag: high/mid/low.
    Loads all rows from the sampled files.
    """
    folder = Path(folder)
    rows = []

    def keyer(f: Path):
        try:
            return int(f.stem.split("_")[-1])
        except Exception:
            return f.stem

    files = sorted(folder.glob("*.txt"), key=keyer)

    # group files by tag
    tag_groups = {"high": [], "mid": [], "low": []}
    for f in files:
        name = f.stem.lower()
        for tag in tag_groups:
            if tag in name:
                tag_groups[tag].append(f)

    # sample 20 files per tag (or all if fewer than 20 exist)
    sampled_files = []
    import random
    for tag, flist in tag_groups.items():
        if flist:
            sampled_files.extend(random.sample(flist, min(n_files, len(flist))))
    print(f"total simulated users: {len(sampled_files)}")
    # load all rows from sampled files
    for i, file in enumerate(sampled_files):
        with open(file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                word, score_str = parts
                try:
                    score = int(score_str)
                except ValueError:
                    continue
                rows.append((i, word, score))

    return pd.DataFrame(rows, columns=["respondent", "word", "score"])

def load_test_folder(folder: str | Path) -> pd.DataFrame:
    """Load test data (CP932), no subsampling. Respondent ID taken from filename."""
    folder = Path(folder)
    rows = []

    # match suffix like "_user123.txt"
    # user_pattern = re.compile(r"user(\d+)\.txt$")
    user_pattern = re.compile(r"(?:user)?(\d+)\.txt$")

    files = sorted(
        [f for f in folder.glob("*.txt") if not f.name.startswith("._")],
        key=lambda f: int(user_pattern.search(f.name).group(1))
    )

    for file in files:
        # extract respondent ID from filename
        match = user_pattern.search(file.name)
        if not match:
            continue
        respondent_id = int(match.group(1))
      
        with open(file, encoding="cp932", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                word, score_str = parts
                try:
                    score = int(score_str)
                except ValueError:
                    continue
                rows.append((respondent_id, word, score))

    return pd.DataFrame(rows, columns=["respondent", "word", "score"])

# ----------------------------
# Stan models
# ----------------------------

def build_stan_code(ordinal: bool) -> str:
    if ordinal:
        return r"""
        data {
          int<lower=1> nwords;
          int<lower=1> nstud;
          int<lower=1> nobs;
          array[nobs] int<lower=1, upper=nstud> stud;
          array[nobs] int<lower=1, upper=nwords> word;
          int<lower=1> ncat;
          array[nobs] int<lower=1, upper=ncat> resp;
        }
        parameters {
          vector[nstud] ability;
          vector<lower=0>[nwords] discriminations;
          vector[nwords] difficulties;
          real<lower=0> d12;
          real<lower=0> d23;
          real<lower=0> d34;
        }
        transformed parameters {
          array[nwords] ordered[4] thresh;
          for (k in 1:nwords) {
            real d = difficulties[k];
            thresh[k,1] = d - d12 - d23 - d34;
            thresh[k,2] = d - d23 - d34;
            thresh[k,3] = d - d34;
            thresh[k,4] = d;
          }
        }
        model {
          ability ~ std_normal();
          difficulties ~ std_normal();
          discriminations ~ normal(1.2, 0.25);
          d12 ~ std_normal();
          d23 ~ std_normal();
          d34 ~ std_normal();
          for (i in 1:nobs) {
            resp[i] ~ ordered_logistic( ability[stud[i]] * discriminations[word[i]],
                                        thresh[word[i]] * discriminations[word[i]] );
          }
        }"""
    else:
        return r"""
        data {
          int<lower=1> nwords;
          int<lower=1> nstud;
          int<lower=1> nobs;
          array[nobs] int<lower=1, upper=nstud> stud;
          array[nobs] int<lower=1, upper=nwords> word;
          array[nobs] int<lower=0, upper=1> resp;
        }
        parameters {
          vector[nstud] ability;
          vector<lower=0>[nwords] discriminations;
          vector[nwords] difficulties;
        }
        model {
          ability ~ std_normal();
          difficulties ~ std_normal();
          discriminations ~ normal(1.2, 0.25);
          for (i in 1:nobs) {
            resp[i] ~ bernoulli_logit(
              (ability[stud[i]] - difficulties[word[i]]) * discriminations[word[i]]
            );
          }
        }"""

def fit_irt(train_df: pd.DataFrame, ordinal: bool, cpp_opts: Dict[str, bool], seed: int = 42) -> pd.DataFrame:
    """Returns DataFrame(word, discrimination, difficulty) learned on train_df."""
    learners = sorted(train_df["respondent"].unique())
    items = sorted(train_df["word"].unique())
    stud_idx = {s: i+1 for i, s in enumerate(learners)}
    item_idx = {w: i+1 for i, w in enumerate(items)}

    if ordinal:
        nobs = len(train_df)
        data = {
            "nwords": len(items),
            "nstud": len(learners),
            "nobs":  nobs,
            "stud":  [stud_idx[s] for s in train_df["respondent"]],
            "word":  [item_idx[w] for w in train_df["word"]],
            "ncat":  5,
            "resp":  train_df["score"].astype(int).tolist(),
        }
    else:
        bin_resp = (train_df["score"].to_numpy() > 3).astype(int)
        data = {
            "nwords": len(items),
            "nstud": len(learners),
            "nobs":  len(train_df),
            "stud":  [stud_idx[s] for s in train_df["respondent"]],
            "word":  [item_idx[w] for w in train_df["word"]],
            "resp":  bin_resp.tolist(),
        }

    stan_code = build_stan_code(ordinal)
    import tempfile
    with tempfile.NamedTemporaryFile("w+", suffix=".stan") as fp:
        fp.write(stan_code)
        fp.flush()
        model = CmdStanModel(stan_file=fp.name, cpp_options=cpp_opts)
        fit = model.optimize(data=data, seed=seed)
    a = np.asarray(fit.stan_variable("discriminations"))
    b = np.asarray(fit.stan_variable("difficulties"))
    return pd.DataFrame({"word": items, "discrimination": a, "difficulty": b})

# ----------------------------
# AB regressor for unseen words (optional)
# ----------------------------

class ABRegressor(nn.Module):
    def __init__(self, dim=300, hidden=300, out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.BatchNorm1d(hidden, momentum=1.0),
            nn.Linear(hidden, out),
        )
    def forward(self, x): return self.net(x)

def load_vectors_txt_or_torchtext(path: str, dim: int = 300) -> Dict[str, np.ndarray]:
    if Vectors is not None and path.endswith(".pt"):
        vecs = Vectors(name=os.path.basename(path).replace(".pt",""), cache=os.path.dirname(path))
        stoi = vecs.stoi
        arr = vecs.vectors.numpy()
        return {w: arr[idx] for w, idx in stoi.items()}
    embs = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip().split()
            if not (len(first) == 2 and first[0].isdigit()):
                f.seek(0)
            for line in tqdm(f, desc="loading embeddings"):
                parts = line.rstrip().split()
                if len(parts) < 301:  # word + 300 dims
                    continue
                w, vals = parts[0], parts[1:]
                try:
                    if len(vals) >= 300:
                        vec = np.asarray(vals[:300], dtype=np.float32)
                        embs[w] = vec
                except Exception:
                    continue
    else:
        warnings.warn(f"Embeddings file not found: {path}; falling back to random.")
    return embs

def train_ab_regressor(train_ab: pd.DataFrame, embed_path: str, dim: int = 300,
                       epochs: int = 100, lr: float = 0.003, bs: int = 128) -> Tuple[ABRegressor, Dict[str, np.ndarray]]:
    words = list(train_ab["word"])
    print("loading word embedding...")
    emb_table = load_vectors_txt_or_torchtext(embed_path, dim)
    print("loading done.")
    X, Y = [], []
    for _, row in train_ab.iterrows():
        w = row["word"]
        vec = emb_table.get(w)
        if vec is None:
            # keep table aligned for re-use later
            vec = np.random.randn(dim).astype(np.float32)
            emb_table[w] = vec
        X.append(vec)
        Y.append([float(row["discrimination"]), float(row["difficulty"])])
    if not X:
        raise RuntimeError("No words to train the AB regressor.")
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X, Y)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ABRegressor(dim, hidden=dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in tqdm(range(epochs), desc="training ab regressor"):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model, emb_table

def predict_ab(words: List[str], model: ABRegressor, emb_table: Dict[str, np.ndarray], dim: int = 300) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = []
    for w in words:
        v = emb_table.get(w)
        if v is None:
            v = np.random.randn(dim).astype(np.float32)
        X.append(v)
    X = torch.tensor(np.stack(X), dtype=torch.float32).to(device)
    with torch.no_grad():
        Y = model(X).cpu().numpy()
    return pd.DataFrame({"word": words, "discrimination": Y[:,0], "difficulty": Y[:,1]})

# ----------------------------
# Item bank + CAT loop
# ----------------------------

@dataclass
class ItemBank:
    words: List[str]
    mat: np.ndarray  # shape (n_items, 4): [a, b, c=0, d=1]

def build_item_bank(ab_df: pd.DataFrame, universe_words: List[str]) -> ItemBank:
    a_map = dict(zip(ab_df["word"], ab_df["discrimination"]))
    b_map = dict(zip(ab_df["word"], ab_df["difficulty"]))
    words = list(universe_words)
    n = len(words)
    mat = np.zeros((n, 4), dtype=float)
    mat[:,2] = 0.0  # c
    mat[:,3] = 1.0  # d
    for i, w in enumerate(words):
        mat[i,0] = float(a_map.get(w, 1.0))
        mat[i,1] = float(b_map.get(w, 0.0))
    return ItemBank(words=words, mat=mat)

from sklearn.linear_model import LogisticRegression

class HillClimbingEstimator(Estimator):
    def __init__(self, use_discriminations=True):
        super().__init__()
        self._use_discriminations = use_discriminations

    def estimate(self, index=None, items=None, administered_items=None, response_vector=None, est_theta=None, **kwargs):
        items, administered_items, response_vector, est_theta = self._prepare_args(
            return_items=True, return_response_vector=True, return_est_theta=True,
            index=index, items=items, administered_items=administered_items,
            response_vector=response_vector, est_theta=est_theta, **kwargs
        )
        if len(set(response_vector)) == 1:
            return cat.dodd(est_theta, items, response_vector[-1])
        log_reg = _fit_log_reg(items, administered_items, response_vector, use_discriminations=self._use_discriminations)
        theta = -log_reg.intercept_[0] / log_reg.coef_[0, 0]
        return float(theta)

def _fit_log_reg(items, administered_items, response_vector, use_discriminations=True, log_reg=None):
    if log_reg is None:
        log_reg = LogisticRegression(C=float("inf"))
    X = items[administered_items][:, 1, np.newaxis]
    sample_weight = items[administered_items, 0] if use_discriminations else None
    log_reg.fit(X, response_vector, sample_weight=sample_weight)
    return log_reg

def select_next(selector_name: str, bank: ItemBank,
                administered: List[int], theta: float,
                rand_helper: Optional[Tuple[List[np.ndarray], Dict[str, int], Dict[str,int]]] = None) -> Optional[int]:
    if selector_name == "max-info":
        sel = MaxInfoSelector()
        return sel.select(items=bank.mat, administered_items=administered, est_theta=theta)
    if selector_name == "urry":
        sel = UrrySelector()
        return sel.select(items=bank.mat, administered_items=administered, est_theta=theta)
    if selector_name == "rand":
        strata, pos_map, counts = rand_helper
        target = 8
        asked = set(administered)
        pool_ids = []
        for s_idx, stratum in enumerate(strata):
            need = counts.get(s_idx, 0) < target
            if need:
                for w in stratum:
                    i = pos_map.get(w)
                    if i is not None and i not in asked:
                        pool_ids.append(i)
        if not pool_ids:
            pool_ids = [i for i in range(len(bank.words)) if i not in asked]
        if not pool_ids:
            return None
        choice = int(np.random.choice(pool_ids))
        for s_idx, stratum in enumerate(strata):
            if bank.words[choice] in set(stratum):
                counts[s_idx] = counts.get(s_idx, 0) + 1
                break
        return choice
    raise ValueError(f"Unknown selector: {selector_name}")

def simulate_cat_one(learner_df: pd.DataFrame,
                     bank: ItemBank,
                     selector: str = "max-info",
                     max_items: int = 40) -> Tuple[float, List[int], List[int], List[float]]:
    gt_map = dict(zip(learner_df["word"], (learner_df["score"].to_numpy() > 3).astype(int)))
    if not gt_map:
        return 0.0, [], [], [0.0]
    rand_helper = None
    if selector == "rand":
        z = word_freq_zscores(bank.words)
        strata = five_strata(bank.words, z)
        pos_map = {w: i for i, w in enumerate(bank.words)}
        counts: Dict[int, int] = {}
        rand_helper = (strata, pos_map, counts)

    est = HillClimbingEstimator()
    stopper = MaxItemStopper(max_items)
    administered, responses, theta_hist = [], [], [0.0]

    for _ in range(max_items):
        nxt = select_next(selector, bank, administered, theta_hist[-1], rand_helper)
        if nxt is None:
            break
        w = bank.words[nxt]
        if w not in gt_map:
            continue
        administered.append(nxt)
        responses.append(int(gt_map[w]))
        theta_new = float(est.estimate(items=bank.mat,
                                       administered_items=administered,
                                       response_vector=responses,
                                       est_theta=theta_hist[-1]))
        theta_hist.append(theta_new)
        if stopper.stop(administered_items=bank.mat[administered], theta=theta_new):
            break

    return theta_hist[-1], administered, responses, theta_hist

def metrics_for_learner(theta: float,
                        ab_df: pd.DataFrame,
                        learner_df: pd.DataFrame,
                        use_discrim_for_pred: bool = True) -> Dict[str, float]:
    eval_words = learner_df["word"].tolist()
    a_map = dict(zip(ab_df["word"], ab_df["discrimination"]))
    b_map = dict(zip(ab_df["word"], ab_df["difficulty"]))
    a = np.array([a_map.get(w, 1.0) for w in eval_words], dtype=float)
    if not use_discrim_for_pred:
        a[:] = 1.0
    b = np.array([b_map.get(w, 0.0) for w in eval_words], dtype=float)
    preds = icc_2pl(theta, a, b)
    y = (learner_df["score"].to_numpy() > 3).astype(int)
    # print(classification_report(y, (preds >= 0.5).astype(int), digits=4))
    return {
        "AUROC": float(roc_auc_score(y, preds)) if len(np.unique(y)) > 1 else float("nan"),
        "MCC": float(matthews_corrcoef(y, (preds >= 0.5).astype(int))) if len(np.unique(y)) > 1 else float("nan"),
        "AP+": float(average_precision_score(y, preds, pos_label=1)),
        "AP-": float(average_precision_score(1 - y, 1 - preds, pos_label=1)),
        "f1":  float(f1_score(y, (preds >= 0.5).astype(int), average="macro")) if len(np.unique(y)) > 1 else float("nan"),
        "acc": float(accuracy_score(y, (preds >= 0.5).astype(int))) if len(np.unique(y)) > 1 else float("nan"),
    }

# ----------------------------
# Orchestration
# ----------------------------

def train_irt_from_folder(train_folder: str | Path, ordinal_irt: bool,
                          use_opencl: bool = True, use_threads: bool = True, n_files: int = 20) -> pd.DataFrame:
    configure_system_threads()
    df_train = load_train_folder(train_folder, n_files=n_files)

    if df_train.empty:
        raise RuntimeError(f"No training data found in {train_folder}")
    cpp_opts = stan_cpp_options(use_opencl=use_opencl, use_threads=use_threads)
    print("fitting irt models...")
    ab = fit_irt(df_train, ordinal=ordinal_irt, cpp_opts=cpp_opts)
    print("irt model fitted")
    return ab

def test_folder_individuals(test_folder: str | Path,
                            ab_train: pd.DataFrame,
                            selector: str = "max-info",
                            max_items: int = 50,
                            use_discrim_for_pred: bool = True,
                            embed_path: Optional[str] = None) -> pd.DataFrame:
    df_test = load_test_folder(test_folder)
    if df_test.empty:
        raise RuntimeError(f"No test data found in {test_folder}")

    # Prepare AB regressor only if needed (unseen words exist & embedding provided)
    model = None
    emb_table = None
    if embed_path is not None:
        train_words = set(ab_train["word"])
        test_words = set(df_test["word"])
        unseen = sorted(list(test_words - train_words))
        print("unseen words num")
        print(len(unseen))
        if unseen:
            model, emb_table = train_ab_regressor(ab_train, embed_path)

    # Evaluate each respondent (file) individually
    results = []
    for rid, ldf in tqdm(df_test.groupby("respondent"), desc = "testing..."):
        eval_words = sorted(ldf["word"].unique())
        # augment AB with predicted params for unseen words if needed
        if model is not None:
            have = set(ab_train["word"])
            missing = [w for w in eval_words if w not in have]
            if missing:
                pred_ab = predict_ab(missing, model, emb_table)
                ab_all = pd.concat([ab_train, pred_ab], ignore_index=True)
            else:
                ab_all = ab_train
        else:
            ab_all = ab_train

        bank = build_item_bank(ab_all, eval_words)
        theta, asked_ids, resp_vec, theta_hist = simulate_cat_one(ldf, bank, selector=selector, max_items=max_items)
        if not asked_ids:
            continue
        m = metrics_for_learner(theta, ab_all, ldf, use_discrim_for_pred=use_discrim_for_pred)
        m.update({"respondent": rid, "n_items": len(asked_ids), "final_theta": theta})
        results.append(m)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Paths
    TRAIN_FOLDER = "path/to/data/qwen_sim_labels/simulated_labels/"  # training txts 
    
    # TEST_FOLDER  = "path/to/data/real_user_labels/vocabulary_knowledge_dataset/"  #  vkd, test every file individually
    TEST_FOLDER =  "path/to/data/evkd_user_labels_five_scale/" # evkd, converted to 5 scale to fit in stan model
    # Optional embeddings (download first):
    # ! wget -O numberbatch.txt.gz https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz
    # ! gzip -d numberbatch.txt.gz  # -> numberbatch.txt
    EMBED_PATH = "path/to/model_checkpoints/numberbatch.txt"


    # Config
    SELECTOR = "max-info"       # "max-info", "urry", or "rand"
    ORDINAL_IRT = False         # False = binary 2PL on score>3; True = graded/ordinal (1..5)
    USE_DISCRIM_FOR_PRED = True
    MAX_ITEMS = 50

    # Train from train/
    ab = train_irt_from_folder(TRAIN_FOLDER, ordinal_irt=ORDINAL_IRT, use_opencl=True, use_threads=True, n_files=50)

    # Test individually 
    metrics_df = test_folder_individuals(
        TEST_FOLDER, ab, selector=SELECTOR, max_items=MAX_ITEMS,
        use_discrim_for_pred=USE_DISCRIM_FOR_PRED, embed_path=EMBED_PATH
    )
    print(metrics_df)
    metrics_df.to_csv("irt_metrics_evkd.csv")

    print("Mean over learners:\n", metrics_df.drop(columns=["respondent"]).mean(numeric_only=True))
