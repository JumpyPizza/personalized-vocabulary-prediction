from __future__ import annotations
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report 
import random

import wandb
import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from train_bert_sent_level_split import (
    apply_fixed_mapping,
    tokenize_and_align_labels,
    train_target_token_classifier,
    ID2LABEL,
    NUM_LABELS,
    ZERO_VALUES,
    
)

import random
import numpy as np
import datasets
from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

import random
import numpy as np
import datasets
from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

def build_user_subset_dataset(
    model_name: str,
    dataset_path: str,
    user_percent: float,       # e.g. 1, 5, 20
    seed: int = 0,
    num_proc: int = 24,
):
   

    # 1) Load per-user datasets
    ds_by_user: DatasetDict = datasets.load_from_disk(dataset_path)
    user_ids = sorted(ds_by_user.keys())
    n_users = len(user_ids)
    if n_users == 0:
        raise ValueError("No user datasets found in the loaded DatasetDict.")

    print(f"Total users = {n_users}")

    # 2) Pick subset of users for training
    n_train_users = max(1, int(n_users * (user_percent / 100.0)))
    random.seed(seed)
    random.shuffle(user_ids)

    train_user_ids = user_ids[:n_train_users]
    val_user_ids = user_ids[n_train_users:]

    print(f"Training users ({len(train_user_ids)}): {train_user_ids[:10]}{' ...' if len(train_user_ids)>10 else ''}")
    print(f"Validation users ({len(val_user_ids)}): {val_user_ids[:10]}{' ...' if len(val_user_ids)>10 else ''}")

    # 3) Concatenate per-user datasets
    def _concat_selected(ids):
        parts = [ds_by_user[uid] for uid in ids if len(ds_by_user[uid]) > 0]
        if not parts:
            raise ValueError("No examples found for selected users.")
        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    train_ds = _concat_selected(train_user_ids)
    val_ds = _concat_selected(val_user_ids)

    print(f"Train examples: {len(train_ds):,} | Validation examples: {len(val_ds):,}")

    # downsampling for speed
    train_ds = train_ds.shuffle(seed=seed).select(range(min(4_000_000, len(train_ds))))
    val_ds = val_ds.shuffle(seed=seed).select(range(min(1_000, len(val_ds))))

    ds_mixed = DatasetDict({"train": train_ds, "validation": val_ds})

    # 4) Apply mapping and tokenization
    print("Applying fixed label mapping...")
    ds_mapped = ds_mixed.map(apply_fixed_mapping, desc="Apply fixed label mapping", num_proc=num_proc)

    model_path = f"path/to/model_checkpoints/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    print("Tokenizing...")
    ds_tok = ds_mapped.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer),
        batched=False,
        num_proc=num_proc,
        desc=f"Tokenizing {model_name}",
    )

    print(f"Final tokenized sizes: train={len(ds_tok['train']):,}, validation={len(ds_tok['validation']):,}")
    return ds_tok, tokenizer



def train_bert_on_user_subset(
    model_name : str,
    dataset_path: str,
    output_root: str,
    fractions=(5, 10, 20, 50),
    seed: int = 0,
):
    """
    Train model on subsets of users: 1%, 5%, 20%.
    Validation comes from 10% of the remaining users.
    """
   
    model_path = f"path/to/model_checkpoints/{model_name}"
    cfg = {"lr": 3e-5, "steps": 3000}

    for frac in fractions:
        print("\n" + "=" * 80)
        print(f"Training on {frac}% of users ")
        print("=" * 80)

        ds_tok, tokenizer = build_user_subset_dataset(
            model_name = model_name,
            dataset_path=dataset_path,
            user_percent=frac,
            seed=seed,
            num_proc=24,
        )

        output_dir = f"{output_root}/bert_{frac}pct_users"
        train_target_token_classifier(
            dataset=ds_tok,
            tokenizer = tokenizer,
            wandb_name=f"bert_base_{frac}pct_users",
            model_name=model_path,
            output_dir=output_dir,
            seed=seed,
            train_batch_size=128,
            eval_batch_size=1028,
            lr=cfg["lr"],
            train_steps=cfg["steps"],
            num_epochs=1,
            warmup_ratio=0.1,
        )


if __name__ == "__main__":
    # ds = datasets.load_from_disk()
    # dataset_path = "path/to/data/qwen_hf_dataset_id"
    # output_dir = f"path/to/trained_checkpoints/multitask_bert_fractions"
    
    dataset_path = "path/to/data/rule_dataset_id"
    output_dir = f"path/to/trained_checkpoints/ablation_rule_bert"
    train_bert_on_user_subset(
        model_name = "bert-base-uncased",
        dataset_path=dataset_path,
        output_root=output_dir,
        # fractions=( 5, 10, 20, 50),
        fractions = [90],
        seed=0,
    )
