#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Any, Tuple, Set
import random
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    classification_report,
)

import datasets
from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)


# =============================
# Label mapping constants
# =============================

LABEL_MAP = {
    0:   -100,   # mask unrelated tokens
    "0": -100,
    "1": 0,      # class 0
    "2": 0,      # class 0
    "3": 0,      # class 0
    "4": 1,      # class 1
    "5": 1,      # class 1
}
ID2LABEL = {0: "L_1_2_3", 1: "L_4_5"}
NUM_LABELS = 2
ZERO_VALUES = {0, "0"}


# =============================
# Helpers: preprocessing
# =============================

def apply_fixed_mapping(example):
    """Map word-level labels using fixed LABEL_MAP and mask others as -100."""
    mapped = []
    for y in example["labels"]:
        if y in ZERO_VALUES:
            mapped.append(-100)
        else:
            key = str(y)
            mapped.append(LABEL_MAP.get(key, -100))
    example["labels"] = mapped
    return example


def tokenize_and_align_labels(example, tokenizer):
    """
    Expand word-level labels to subword-level:
    - first subword keeps label
    - subsequent subwords → -100
    - special tokens → -100
    """
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=256,
    )
    word_ids = tokenized.word_ids()
    word_level = example["labels"]

    labels = []
    prev = None
    for wi in word_ids:
        if wi is None:
            labels.append(-100)
        else:
            if wi != prev:
                labels.append(word_level[wi])
            else:
                labels.append(-100)
        prev = wi

    tokenized["labels"] = labels
    return tokenized


# =============================
# Metrics
# =============================

def compute_metrics(eval_pred):
    """Compute accuracy, macro-f1, and MCC on valid tokens only."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    y_true, y_pred = [], []
    for p, l in zip(preds, labels):
        mask = l != -100
        if mask.any():
            y_true.extend(l[mask].tolist())
            y_pred.extend(p[mask].tolist())

    if not y_true:
        return {"accuracy": 0.0, "f1": 0.0, "mcc": 0.0}

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def _y_true_pred_from_logits(logits, labels):
    preds = logits.argmax(-1)
    y_true, y_pred = [], []
    for p_seq, l_seq in zip(preds, labels):
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            y_true.append(int(l))
            y_pred.append(int(p))
    return np.array(y_true), np.array(y_pred)


# =============================
# Word-disjoint split
# =============================

def _split_user_by_word_disjoint(
    ds: Dataset,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> Tuple[Dataset, Dataset]:
    """
    Split a single user's dataset into train/test by disjoint `word` values.
    """
    words = [w for w in ds.unique("word") if isinstance(w, str) and w.strip()]
    if len(words) < 2:
        return Dataset.from_dict({k: [] for k in ds.features}), Dataset.from_dict({k: [] for k in ds.features})

    rng = random.Random(seed)
    rng.shuffle(words)

    n_train = int(round(len(words) * train_ratio))
    if n_train == 0:
        n_train = 1
    if n_train == len(words):
        n_train = len(words) - 1

    train_words = set(words[:n_train])
    test_words = set(words[n_train:])

    train_ds = ds.filter(lambda w: w in train_words, input_columns=["word"])
    test_ds  = ds.filter(lambda w: w in test_words,  input_columns=["word"])

    assert (set(train_ds.unique("word")) & set(test_ds.unique("word"))) == set(), "Word sets overlapped!"
    return train_ds, test_ds


# =============================
# Main training loop
# =============================

def intra_user_word_split_eval_dataset(
    dataset_path: str,
    model_path: str,
    seed: int = 0,
    train_batch_size: int = 128,
    eval_batch_size: int = 256,
    num_epochs: float = 1.0,
    lr: float = 5e-5,
    num_proc: int = 24,
    train_ratio: float = 0.8,
    eval_steps: int = 500,
):
    """
    Train & evaluate within each user:
    - split by word disjointness (80/20)
    - no model saving
    - eval every `eval_steps`
    - collect metrics per user
    """
    set_seed(seed)
    ds_by_user: DatasetDict = datasets.load_from_disk(dataset_path)
    user_ids = list(ds_by_user.keys())
    print(f"Users loaded: {len(user_ids)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    label2id = {v: k for k, v in ID2LABEL.items()}

    all_results = {}

    for uid in user_ids:
        print("\n" + "="*70)
        print(f"Intra-user 80/20 word-disjoint: user = {uid}")
        print("="*70)

        ds_user = ds_by_user[uid]
        if len(ds_user) == 0:
            print(f"[skip] no examples for {uid}")
            continue

        # 1) Split
        raw_train_ds, raw_test_ds = _split_user_by_word_disjoint(
            ds_user, train_ratio=train_ratio, seed=hash((seed, uid)) & 0xffffffff
        )
        if len(raw_train_ds) == 0 or len(raw_test_ds) == 0:
            print(f"[skip] empty split for {uid}")
            continue

        # 2) Map + tokenize
        mapped_train = raw_train_ds.map(apply_fixed_mapping, desc=f"map[{uid}::train]", num_proc=num_proc)
        mapped_test  = raw_test_ds.map(apply_fixed_mapping,  desc=f"map[{uid}::test]",  num_proc=num_proc)

        tok_train = mapped_train.map(
            lambda ex: tokenize_and_align_labels(ex, tokenizer),
            batched=False, desc=f"tok[{uid}::train]", num_proc=num_proc
        )
        tok_test  = mapped_test.map(
            lambda ex: tokenize_and_align_labels(ex, tokenizer),
            batched=False, desc=f"tok[{uid}::test]", num_proc=num_proc
        )

        # 3) Model
        config = AutoConfig.from_pretrained(
            model_path, num_labels=NUM_LABELS,
            id2label=ID2LABEL, label2id=label2id
        )
        model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
        collator = DataCollatorForTokenClassification(tokenizer)

        args = TrainingArguments(
            output_dir="./_tmp",  # required, but unused
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=lr,
            num_train_epochs=num_epochs,
            logging_strategy="steps",
            logging_steps=eval_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="no",   # do not save checkpoints
            report_to=[],
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tok_train,
            eval_dataset=tok_test,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        # 4) Train + evaluate
        trainer.train()
        preds = trainer.predict(tok_test)
        y_true, y_pred = _y_true_pred_from_logits(preds.predictions, preds.label_ids)

        # summary metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        # classification report dict
        target_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_LABELS)),
            target_names=target_names,
            digits=4,
            zero_division=0,
            output_dict=True,
        )

        results = {
            "accuracy": acc,
            "macro_f1": f1,
            "mcc": mcc,
            "report": report_dict,
        }

       
        all_results[uid] = results
        print(f"User {uid} results: f1: {f1}")

    # Summary
    print("\n========== Summary across users ==========")
    for uid, m in all_results.items():
        print(f"{uid}: acc={m['accuracy']:.4f}, f1={m['macro_f1']:.4f}, mcc={m['mcc']:.4f}")

    return all_results


# =============================
# Example entry point
# =============================

if __name__ == "__main__":
    dataset_path = "path/to/data/real_user_hf_dataset"
    model_path = "path/to/model_checkpoints/bert-base-uncased"

    results = intra_user_word_split_eval_dataset(
        dataset_path=dataset_path,
        model_path=model_path,
        seed=0,
        train_batch_size=64,
        eval_batch_size=128,
        num_epochs=1.0,
        lr=5e-5,
        num_proc=24,
        train_ratio=0.8,
        eval_steps=2000,
    )
    import json
    from pathlib import Path

    # inside intra_user_word_split_eval_dataset (after printing summary):
    out_file = Path("./self_partition_results.json")  # change path as needed
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved detailed results to {out_file}")
