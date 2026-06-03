
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Union
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import json
import random
import numpy as np
import pandas as pd

import datasets
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import matthews_corrcoef as mcc
from tqdm import tqdm 

import datasets
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from frequency_sampling import stratified_sample_by_frequency

# ---- fixed mapping from earlier ----
LABEL_MAP = {0:-100, "0": -100, "1": 0, "2": 0, "3": 0, "4": 1, "5": 1}
ID2LABEL = {0: "L_1_2_3", 1: "L_4_5"}
NUM_LABELS = 2
ZERO_VALUES = {0, "0"}





def read_user_jsonl(path: Union[str, Path]) -> Dict[str, List[Dict]]:
    """
    Each line: { "<user>": [ { "word": w, "tokens": [[...],...], "labels": [[...],...] }, ... ] }
    Returns: {user -> list[items]}
    """

    users = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f)):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list):           # no username wrapper
                users[idx] = obj
            elif isinstance(obj, dict):         # { user: [...] }
                (user, payload), = obj.items()
                users[idx] = payload
            else:
                raise RuntimeError("Incorrect JSONL line format.")
    return users

def apply_fixed_mapping_word_level(labels: List[Union[int, str]]) -> List[int]:
    """
    Map a single sentence's word-level labels using LABEL_MAP, mask zeros to -100.
    Unknown labels default to -100 (ignored).
    """
    out = []
    for y in labels:
        if y in ZERO_VALUES:
            out.append(-100)
        else:
            key = str(y).lstrip().rstrip()
            out.append(LABEL_MAP.get(key, -100))
    return out

def tokenize_and_align_labels(tokens: List[str], labels: List[int], tokenizer, max_length=256) -> Dict:
    """
    tokens: word-level tokens
    labels: word-level labels already mapped to {-100, 0, 1}
    Returns tokenized inputs with subword-aligned labels:
      - first subword gets the label
      - continuation subwords and special tokens get -100
    """
    tokenized = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=max_length)
    word_ids = tokenized.word_ids()

    out_labels = []
    prev_wid = None
    for wid in word_ids:
        if wid is None:
            out_labels.append(-100)
        else:
            if wid != prev_wid:
                out_labels.append(labels[wid])
            else:
                out_labels.append(-100)
        prev_wid = wid

    tokenized["labels"] = out_labels
    return tokenized


def compute_metrics(eval_pred):
    """Evaluate ONLY where gold label != -100 (i.e., target token positions)."""

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    y_true, y_pred = [], []
    for p, l in zip(preds, labels):
        mask = l != -100
        if mask.any():
            y_true.extend(l[mask].tolist())
            y_pred.extend(p[mask].tolist())

    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": mcc(y_true, y_pred)
    }


def _masked_classification_report(trainer: Trainer, eval_ds: Dataset, title: str):
    """
    Runs prediction and prints a classification report using only positions where label != -100.
    """
    preds_output = trainer.predict(eval_ds)
    logits = preds_output.predictions
    labels = preds_output.label_ids

    y_pred = np.argmax(logits, axis=-1)
    true, pred = [], []
    for p, l in zip(y_pred, labels):
        mask = l != -100
        if mask.any():
            true.extend(l[mask].tolist())
            pred.extend(p[mask].tolist())

    if not true:
        print(f"{title}: No labeled tokens (after masking) to evaluate.")
        return

    # Print a nice report (per class + macro/weighted)
    print(f"\n==== {title} ====")
    print(classification_report(true, pred, target_names=[ID2LABEL[i] for i in sorted(set(true))], zero_division=0))
    # Also print the compact metrics you were tracking, if you like:
    acc = accuracy_score(true, pred)
    prec = precision_score(true, pred, average="macro", zero_division=0)
    rec = recall_score(true, pred, average="macro", zero_division=0)
    f1 = f1_score(true, pred, average="macro", zero_division=0)
    mcc_score = mcc(true, pred)
    print(f"(macro) accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}. mcc = {mcc_score:.4f}\n")


@dataclass
class FinetuneConfig:
    base_ckpt_path: str
    output_dir: str
    n_words_train: int = 30 # how many words as training data
    seed: int = 0
    train_batch_size: int = 16 # per device train batch size
    eval_batch_size: int = 32
    lr: float = 5e-5
    num_epochs: float = 1.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    max_length: int = 256


def _build_user_datasets_for_words(
    items: List[Dict], train_words: set, test_words: set, tokenizer, max_length: int
) -> Tuple[Dataset, Dataset]:
    """
    From one user's items -> build sentence-level HF datasets for train (train_words) and test (test_words).
    Each item has: word, tokens: [[...], ...], labels: [[...], ...]
    """
    def collect_for(target_set: set, set:  str) -> List[Dict]:
        exs: List[Dict] = []
        for it in items:
            w = it.get("word")
   
            if w not in target_set:
                continue
            for idx, (toks, labs) in enumerate(zip(it.get("tokens", []) or [], it.get("labels", []) or [])):
                if set == "train":
                    if idx > 5: #  number of sentence samples per label for training
                        break
                if set == "test":
                    if idx > 2:  #  number of sentence samples per label for testing
                        break
                # map word-level labels, drop sentences where mapped labels are all -100
                mapped = apply_fixed_mapping_word_level(labs)
                if all(l == -100 for l in mapped):
                    continue
                tok_out = tokenize_and_align_labels(toks, mapped, tokenizer, max_length)
                exs.append(tok_out)
   
        return exs


    train_examples = collect_for(train_words, "train")
    test_examples  = collect_for(test_words, "test")

    if not train_examples:
        raise ValueError("User has no train examples after mapping. Reduce n_words_train or check data.")
    if not test_examples:
        raise ValueError("User has no test examples after mapping. Reduce n_words_train or check data.")

    return Dataset.from_list(train_examples), Dataset.from_list(test_examples)


def finetune_per_user(
    jsonl_path: Union[str, Path],
    cfg: FinetuneConfig,
    model_name: None
) -> Dict[str, Dict]:
    """
    For each user (independently):
      - sample N words for fine-tuning, rest for testing
      - load the SAME base checkpoint (cfg.base_ckpt_path)
      - print classification report BEFORE fine-tuning (on that user's test words)
      - fine-tune
      - print classification report AFTER fine-tuning (same test set)
      - do NOT save the fine-tuned checkpoint
    """
    set_seed(cfg.seed) 
    users_data = read_user_jsonl(jsonl_path)

    # tokenizer from the same checkpoint
    if "roberta" in cfg.base_ckpt_path:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_ckpt_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_ckpt_path, use_fast=True)

    results: Dict[str, Dict] = {"users": {}}
    all_f1, all_acc, all_mcc = [], [], []

    for user, items in tqdm(users_data.items(), desc="Users"):
       
        unique_words = sorted({it.get("word") for it in items if it.get("word") is not None})
        if len(unique_words) < 2:
            raise RuntimeError("no words for training")   # need at least 1 train word + 1 test word
       
        rnd = random.Random(cfg.seed + hash(user) % (2**16))
        n_train = min(cfg.n_words_train, max(1, len(unique_words) - 1))
        train_words = set(rnd.sample(unique_words, n_train)) 
        test_words  = set(unique_words) - train_words
        if not test_words:
            w_move = rnd.choice(list(train_words))
            train_words.remove(w_move)
            test_words.add(w_move)

        train_ds, test_ds = _build_user_datasets_for_words(items, train_words, test_words, tokenizer, cfg.max_length)

        # Fresh model from the SAME checkpoint for this user
        config = AutoConfig.from_pretrained(
            cfg.base_ckpt_path,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id={v: k for k, v in ID2LABEL.items()},
        )
        model = AutoModelForTokenClassification.from_pretrained(cfg.base_ckpt_path, config=config)

        collator = DataCollatorForTokenClassification(tokenizer)
        out_dir_user = Path(cfg.output_dir) / f"user_{user}"
        # out_dir_user.mkdir(parents=True, exist_ok=True)

        # Do NOT save checkpoints; just evaluate and print reports
        args = TrainingArguments(
            output_dir=str(out_dir_user),
            per_device_train_batch_size=cfg.train_batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            learning_rate=cfg.lr,
            num_train_epochs=cfg.num_epochs,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            eval_strategy="no",  # we will call evaluate/predict explicitly
            save_strategy="no",        # <-- no saving
            load_best_model_at_end=False,
            report_to="none",
            seed=cfg.seed,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,  # macro metrics over masked positions
        )

        # --- Evaluate BEFORE fine-tuning ---
        # _masked_classification_report(trainer, test_ds, title=f"User {user} — BEFORE fine-tuning")

        # --- Fine-tune ---
        trainer.train()

        # --- Evaluate AFTER fine-tuning ---
        metrics = trainer.evaluate()
        # _masked_classification_report(trainer, test_ds, title=f"User {user} — AFTER fine-tuning")

        # Record compact metrics (macro) for aggregation
        
        all_f1.append(metrics["eval_f1"])
        
        all_acc.append(metrics["eval_accuracy"])
        all_mcc.append(metrics["eval_mcc"])

        results["users"][user] = {
            "n_train_words": len(train_words),
            "n_test_words": len(test_words),
            "metrics_after": metrics,  # compact macro metrics from compute_metrics
        }

        print(metrics["eval_f1"])

    # Aggregate (macro) across users
    results["macro_f1"] = float(np.mean(all_f1)) if all_f1 else 0.0
    results["macro_acc"] = float(np.mean(all_acc)) if all_acc else 0.0
    results["average_mcc"] = float(np.mean(all_mcc)) if all_mcc else 0.0
    if model_name:
        row = {
            "user_id":[],
            "f1_macro":[],
            "acc":[],
            "mcc":[],
        }
        for user_id, data in results["users"].items():
            f1 = data["metrics_after"]["eval_f1"]
            acc = data["metrics_after"]["eval_accuracy"]
            mcc = data["metrics_after"]["eval_mcc"]
            row["user_id"].append(user_id)
            row["f1_macro"].append(f1)
            row["acc"].append(acc)
            row["mcc"].append(mcc)
        pd.DataFrame.from_dict(row).to_csv(f"./{model_name}.csv", index=False)
    return results

if __name__ == "__main__":
    lr_param = {
            "bert-base" : 3e-5, 
            "bert-large-uncased" : 5e-6,
            "roberta-base" : 1e-5, 
            "roberta-large": 5e-6, 
            "deberta-v3-base" : 1e-5, 
            "deberta-v3-large" : 5e-6


        }
    
    # user_path = "path/to/data/sentence_level_real_user_labels.json"
    # user_path = "path/to/data/sentence_level_evkd_user_labels.jsonl"
    # ckpt_path = "path/to/trained_checkpoints/full_finetune_bert"
    # for model_name in ["bert-base", "bert-large-uncased", "roberta-base", "roberta-large", "deberta-v3-base", "deberta-v3-large"]:
    # for model_name in [ "deberta-v3-large"]:
    #     ckpt_path = f"path/to/trained_checkpoints/multitask_{model_name}"
        
    #     # run_num = 5
    #     # for run in range(run_num):
    #     cfg = FinetuneConfig(base_ckpt_path = ckpt_path, output_dir = "./output", lr = lr_param[model_name], 
    #                         n_words_train = 50, train_batch_size = 50, eval_batch_size = 128, num_epochs = 1)
    #     results = finetune_per_user(user_path, cfg, model_name = model_name)
      

 

    # Example user path
    # user_path = "path/to/data/sentence_level_real_user_labels.json"
    # run_num = 5  # how many independent runs per model

    # results_summary = []
  
    # all_user_stats = []
    # for model_name in [
    #     "bert-base-uncased",
    #     # "bert-large-uncased",
    #     # "roberta-base",
    #     # "roberta-large",
    #     # "deberta-v3-base",
    #     # "deberta-v3-large",
    # ]:
    #     ckpt_path = f"path/to/trained_checkpoints/multitask_{model_name}"

    #     run_f1, run_acc, run_mcc = [], [], []
    #     user_scores = defaultdict(lambda: {"f1": [], "acc": [], "mcc": []})
    #     for run in range(run_num):
    #         seed = 0 + run
    #         print(f"\n===== Running {model_name}, run {run + 1}/{run_num} (seed={seed}) =====")

    #         cfg = FinetuneConfig(
    #             base_ckpt_path=ckpt_path,
    #             output_dir=f"./output",
    #             lr=lr_param[model_name],
    #             n_words_train=50,
    #             train_batch_size=50,
    #             eval_batch_size=1024,
    #             num_epochs=1,
    #             seed=seed,
    #         )

    #         results = finetune_per_user(user_path, cfg, model_name=None)

    #         for user_id, data in results["users"].items():
    #             user_scores[user_id]["f1"].append(data["metrics_after"]["eval_f1"])
    #             user_scores[user_id]["acc"].append(data["metrics_after"]["eval_accuracy"])
    #             user_scores[user_id]["mcc"].append(data["metrics_after"]["eval_mcc"])

    #     # compute per-user averages & stds
    #     per_user_rows = []
    #     for user_id, vals in user_scores.items():
    #         per_user_rows.append({
    #             "model": model_name,
    #             "user_id": user_id,
    #             "f1_mean": np.mean(vals["f1"]),
    #             "f1_std": np.std(vals["f1"]),
    #             "acc_mean": np.mean(vals["acc"]),
    #             "acc_std": np.std(vals["acc"]),
    #             "mcc_mean": np.mean(vals["mcc"]),
    #             "mcc_std": np.std(vals["mcc"]),
    #         })
    #     all_user_stats.extend(per_user_rows)

    #     # compute model-level averages (mean across users)
    #     df_users = pd.DataFrame(per_user_rows)
    #     model_summary = {
    #         "model": model_name,
    #         "f1_mean_over_users": df_users["f1_mean"].mean(),
    #         "f1_std_over_users": df_users["f1_mean"].std(),
    #         "acc_mean_over_users": df_users["acc_mean"].mean(),
    #         "acc_std_over_users": df_users["acc_mean"].std(),
    #         "mcc_mean_over_users": df_users["mcc_mean"].mean(),
    #         "mcc_std_over_users": df_users["mcc_mean"].std(),
    #     }
    #     results_summary.append(model_summary)

    #     print(f"\n===== {model_name} summary across users =====")
    #     for k, v in model_summary.items():
    #         if k != "model":
    #             print(f"{k}: {v:.4f}")

    # # Save per-user and per-model results
    # pd.DataFrame(all_user_stats).to_csv("./user_stats_by_model.csv", index=False)
    # pd.DataFrame(results_summary).to_csv("./model_run_summary.csv", index=False)
  

    # user_path = "path/to/data/sentence_level_real_user_labels.json"
    user_path = "path/to/data/sentence_level_evkd_user_labels.jsonl"

    results_summary = []
  
    all_user_stats = []
    model_name = "deberta-v3-base"
    
    seed = 0
    ckpt_path = f"path/to/trained_checkpoints/multitask_evkd_deberta-v3-base/"
    
    run_f1, run_acc, run_mcc = [], [], []
    user_scores = defaultdict(lambda: {"f1": [], "acc": [], "mcc": []})
    

    cfg = FinetuneConfig(
        base_ckpt_path=ckpt_path,
        output_dir=f"./output",
        lr=lr_param[model_name],
        n_words_train=50,
        train_batch_size=50,
        eval_batch_size=1024,
        num_epochs=1,
        seed=seed,
    )

    results = finetune_per_user(user_path, cfg, model_name=None)

    for user_id, data in results["users"].items():
        user_scores[user_id]["f1"].append(data["metrics_after"]["eval_f1"])
        user_scores[user_id]["acc"].append(data["metrics_after"]["eval_accuracy"])
        user_scores[user_id]["mcc"].append(data["metrics_after"]["eval_mcc"])

    per_user_rows = []
    for user_id, vals in user_scores.items():
        per_user_rows.append({
            "model": model_name,
            "user_id": user_id,
            "f1_mean": np.mean(vals["f1"]),
        
            "acc_mean": np.mean(vals["acc"]),
        
            "mcc_mean": np.mean(vals["mcc"]),
        
        })



    df_users = pd.DataFrame(per_user_rows)
    model_summary = {
        "model": model_name,
        "f1_mean_over_users": df_users["f1_mean"].mean(),
    
        "acc_mean_over_users": df_users["acc_mean"].mean(),
    
        "mcc_mean_over_users": df_users["mcc_mean"].mean(),
    }
    results_summary.append(model_summary)
    df_users.to_csv(f"./deberta_evkd.csv", index=False)
    for k, v in model_summary.items():
        if k != "model":
            print(f"{k}: {v}")

    
