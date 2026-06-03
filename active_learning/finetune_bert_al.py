
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Union, Optional
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch
import datasets
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import matthews_corrcoef as mcc


from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from active_learning.base import StrategyConfig, use_strategy
from active_learning.coreset import sequence_embeddings

# Import strategy modules so their @register_strategy decorators populate the registry.
import active_learning.bald
import active_learning.coreset
import active_learning.eer
import active_learning.uncertainty

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



class ActiveRunner:
    def __init__(self, strategy_name: str, tokenizer, max_length: int, device: str = "cuda"):
        self.strategy_name = strategy_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def select_words(self, model, test_ds, select_k: int, batch_size: int = 128, seq_agg="mean", word_agg="mean", mc_samples: int = 0):
        cfg = StrategyConfig(
            batch_size=batch_size,
            max_length=self.max_length,
            device=self.device,
            seq_agg=seq_agg,
            word_agg=word_agg,
            mc_samples=mc_samples,
        )
        strategy = use_strategy(self.strategy_name, cfg)
        if len(test_ds) == 0 or select_k <= 0:
            return []
        return strategy.select_words(model=model, tokenizer=self.tokenizer, dataset=test_ds, select_k=select_k)

    def evaluate(self, model, test_ds, collator, compute_metrics, output_dir, seed=0, eval_bs=256):
        if len(test_ds) == 0:
            return {"eval_accuracy": 0.0, "eval_precision": 0.0, "eval_recall": 0.0, "eval_f1": 0.0, "eval_mcc": 0.0}
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=eval_bs,
            do_train=False,
            do_eval=True,
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            seed=seed,
        )
        eval_trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=test_ds,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        return eval_trainer.evaluate()




# ---- 
# preserves meta_word + masks for AL scoring 
# ----
def _build_user_datasets_for_words_with_meta(
    items: List[Dict],
    train_words: set,
    test_words: set,
    tokenizer,
    max_length: int,
    train_per_word: int = 6,  # up to N sentences per word for train
    test_per_word: int = 3,   # up to N sentences per word for test
) -> Tuple[Dataset, Dataset]:
    """
    Same idea as your _build_user_datasets_for_words but:
      - attaches 'meta_word' for each example so AL can aggregate per word
      - adds 'special_tokens_mask' used to exclude [CLS]/[SEP] from scoring
    """
    def collect(target_words: set, split: str) -> List[Dict]:
        exs: List[Dict] = []
        for it in items:
            w = it.get("word")
            if w not in target_words:
                continue
            caps = train_per_word if split == "train" else test_per_word
            taken = 0
            for toks, labs in zip(it.get("tokens", []) or [], it.get("labels", []) or []):
                if taken >= caps:
                    break
                mapped = apply_fixed_mapping_word_level(labs)
                if all(l == -100 for l in mapped):
                    continue
                # tokenize with masks
                tok = tokenizer(
                    toks,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=max_length,
                    return_special_tokens_mask=True,
                )
                word_ids = tok.word_ids()
                out_labels = []
                prev = None
                for wid in word_ids:
                    if wid is None:
                        out_labels.append(-100)
                    else:
                        if wid != prev:
                            out_labels.append(mapped[wid])
                        else:
                            out_labels.append(-100)
                        prev = wid
                tok["labels"] = out_labels
                tok["meta_word"] = w
                exs.append(tok)
                taken += 1
        return exs

    train_examples = collect(train_words, "train") if train_words else []
    test_examples  = collect(test_words,  "test")  if test_words  else []

    train_ds = Dataset.from_list(train_examples) if train_examples else Dataset.from_list([])
    test_ds  = Dataset.from_list(test_examples)  if test_examples  else Dataset.from_list([])

    return train_ds, test_ds




@dataclass
class FinetuneConfig:
    base_ckpt_path: str
    output_dir: str
    n_words_train: int = 50 # $N$
    seed: int = 0
    train_batch_size: int = 50 # per device train batch size
    eval_batch_size: int = 1024
    lr: float = 5e-5
    num_epochs: float = 1.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    max_length: int = 1024


def finetune_per_user(
    jsonl_path: Union[str, Path],
    cfg: FinetuneConfig,
    model_name: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Active Learning version.
    For each user:
      - build initial test set over ALL words
      - iterate: select_k words via AL, add to train, (re)train, shrink unlabeled pool
      - final evaluate on remaining test set (or keep a held-out test split if desired)
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

    # AL config
    select_k = getattr(cfg, "al_select_k", 5)                # words per round
    strategy_name = getattr(cfg, "al_strategy", "uncertainty_entropy")
    mc_samples = getattr(cfg, "al_mc_samples", 0)
    seq_agg = getattr(cfg, "al_seq_agg", "mean")
    word_agg = getattr(cfg, "al_word_agg", "mean")
    al_eval_bs = getattr(cfg, "eval_batch_size", cfg.eval_batch_size)

    for user, items in tqdm(users_data.items(), desc="Users"):
        # ---------- build word space ----------
        unique_words = sorted({it.get("word") for it in items if it.get("word") is not None})
        if len(unique_words) < 2:
            raise RuntimeError("no words for training")  # need at least 1 train word + 1 for testing

        train_words: set = set()
        test_words:  set = set(unique_words)  # start: everything is unlabeled

        # ---------- init model per user ----------
        config = AutoConfig.from_pretrained(
            cfg.base_ckpt_path,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id={v: k for k, v in ID2LABEL.items()},
        )
        model = AutoModelForTokenClassification.from_pretrained(cfg.base_ckpt_path, config=config)

        collator = DataCollatorForTokenClassification(tokenizer)
        out_dir_user = Path(cfg.output_dir) / f"user_{user}"

        # ---------- build initial pools ----------
        _, test_ds = _build_user_datasets_for_words_with_meta(
            items,
            train_words=train_words,
            test_words=test_words,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            train_per_word=6,
            test_per_word=3,
        )
        setattr(test_ds, "labeled_bank", None) # diverse sampling

        runner = ActiveRunner(
            strategy_name=strategy_name,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            device="cuda"
        )

        # ---------- AL loop ----------
        budget = min(cfg.n_words_train, max(1, len(unique_words) - 1))
        rounds = max(1, int(np.ceil(budget / select_k)))

        for r in range(rounds):
            # pick how many words this round
            remaining_budget = budget - len(train_words)
            if remaining_budget <= 0 or len(test_words) == 0:
                break
            k_this = min(select_k, remaining_budget, len(test_words))

            # 1) SELECT words via strategy run over current test_ds
            chosen_words = runner.select_words(
                model=model,
                test_ds=test_ds,
                select_k=k_this,
                batch_size=al_eval_bs,
                seq_agg=seq_agg,
                word_agg=word_agg,
                mc_samples=mc_samples
            )

            if not chosen_words:   
                raise RuntimeError("no chosen words")
            

            # 2) MOVE selected words from unlabeled -> train
            train_words.update(chosen_words)
            test_words.difference_update(chosen_words)

            # 3) REBUILD train/test datasets
            train_ds, test_ds = _build_user_datasets_for_words_with_meta(
                items,
                train_words=train_words,
                test_words=test_words,
                tokenizer=tokenizer,
                max_length=cfg.max_length,
                train_per_word=6,
                test_per_word=3,
            )

            # 4) TRAIN on current train_ds
            if len(train_ds) > 0:
                
                L_vecs, L_words = sequence_embeddings(model, train_ds, torch.device("cuda"), batch_size=cfg.eval_batch_size)
                # average per word
                L_map = {}
                for v, w in zip(L_vecs, L_words):
                    L_map.setdefault(w, []).append(v)
                labeled_bank = np.stack([np.mean(L_map[w], axis=0) for w in L_map.keys()], axis=0)
                setattr(test_ds, "labeled_bank", labeled_bank)
     


                args = TrainingArguments(
                    output_dir=str(out_dir_user),
                    per_device_train_batch_size=cfg.train_batch_size,
                    per_device_eval_batch_size=cfg.eval_batch_size,
                    learning_rate=cfg.lr,
                    num_train_epochs=cfg.num_epochs,
                    weight_decay=cfg.weight_decay,
                    warmup_ratio=cfg.warmup_ratio,
                    logging_steps=cfg.logging_steps,
                    evaluation_strategy="no",
                    save_strategy="no",
                    load_best_model_at_end=False,
                    report_to="none",
                    seed=cfg.seed,
                )
                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_ds,
                    eval_dataset=None,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    compute_metrics=compute_metrics,
                )
                trainer.train()

            # interim evaluation each round:
            # interim_metrics = runner.evaluate(model, test_ds, collator, compute_metrics, str(out_dir_user), seed=cfg.seed, eval_bs=al_eval_bs)
            # print(f"[User {user}] Round {r+1}/{rounds} interim F1: {interim_metrics.get('eval_f1', 0):.4f}")

      
        final_metrics = runner.evaluate(model, test_ds, collator, compute_metrics, str(out_dir_user), seed=cfg.seed, eval_bs=al_eval_bs)

        all_f1.append(final_metrics.get("eval_f1", 0.0))
        all_acc.append(final_metrics.get("eval_accuracy", 0.0))
        all_mcc.append(final_metrics.get("eval_mcc", 0.0))

        results["users"][user] = {
            "n_train_words": len(train_words),
            "n_test_words": len(test_words),
            "metrics_after": final_metrics,
        }

        print(final_metrics.get("eval_f1", 0.0))

    # Aggregate across users
    results["macro_f1"] = float(np.mean(all_f1)) if all_f1 else 0.0
    results["macro_acc"] = float(np.mean(all_acc)) if all_acc else 0.0
    results["average_mcc"] = float(np.mean(all_mcc)) if all_mcc else 0.0

    # enable record by passing model_name
    if model_name:
        row = {"user_id": [], "f1_macro": [], "acc": [], "mcc": []}
        for user_id, data in results["users"].items():
            row["user_id"].append(user_id)
            row["f1_macro"].append(data["metrics_after"].get("eval_f1", 0.0))
            row["acc"].append(data["metrics_after"].get("eval_accuracy", 0.0))
            row["mcc"].append(data["metrics_after"].get("eval_mcc", 0.0))
        pd.DataFrame.from_dict(row).to_csv(f"./{model_name}.csv", index=False)

    return results
