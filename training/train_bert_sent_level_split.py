from __future__ import annotations
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report 




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

ZERO_VALUES = {0, "0"}  # treat these as unrelated → -100

def apply_fixed_mapping(example):
    """Map word-level labels using fixed LABEL_MAP and mask others as -100."""
    mapped = []
    for y in example["labels"]:
        if y in ZERO_VALUES:          # unrelated token
            mapped.append(-100)
        else:
            key = str(y)              # labels are given as strings "1".."5" per spec
            mapped.append(LABEL_MAP.get(key, -100))  # unknowns → masked
    example["labels"] = mapped
    return example

def tokenize_and_align_labels(example, tokenizer):
    """
    We already have word-level tokens. Expand labels to subwords:
    - first subword keeps label ( -100)
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

    # word-level labels already mapped to {-100, 0, 1}
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
                labels.append(-100)   # mask continuation pieces
        prev = wi

    tokenized["labels"] = labels
    return tokenized

def compute_metrics(eval_pred):
    """Evaluate only where gold label != -100 (target-token positions)."""
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
    }



def train_target_token_classifier(
    dataset,
    tokenizer,
    wandb_name: str,
    model_name: str,
    output_dir: str,
    seed: int = 0,
    train_steps: int = 3000,
    train_batch_size: int = 128,
    eval_batch_size: int = 256,
    lr: float = 5e-5,
    num_epochs: float = 1.0,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.01,
    logging_steps: int = 50,
    report_to = "wandb"
):
    """
    Dataset must have splits and fields:
      - "tokens": List[str]
      - "labels": List[Union[str,int]] where the target token has label in {"1","2","3","4","5"}
                  and unrelated tokens are 0 / "0".
    """
    set_seed(seed)
    if report_to == "wandb":
        wandb.init(project="vocab-predict-full_training-qwen", name=wandb_name)

    # 3) Model/config
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},  # inverse: {"L_1_2_3":0, "L_4_5":1}
    )
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

    collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=lr,
        max_steps=train_steps,  
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to = report_to,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("Eval:", eval_metrics)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if report_to == "wandb":
        wandb.finish()
    return {"metrics": eval_metrics}



######### code for LOO test ############################
def _concat_except(ds_by_user: DatasetDict, excluded_user: str) -> Dataset:
    parts = [ds for uid, ds in ds_by_user.items() if uid != excluded_user and len(ds) > 0]
    if not parts:
        raise ValueError("Empty train set after excluding user:", excluded_user)
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

def _y_true_pred_from_logits(logits, labels):
    preds = logits.argmax(-1)
    y_true, y_pred = [], []
    for p_seq, l_seq in zip(preds, labels):
        for p, l in zip(p_seq, l_seq):
            if l == -100:  # ignore masked tokens
                continue
            y_true.append(int(l))
            y_pred.append(int(p))
    return np.array(y_true), np.array(y_pred)

def loo_eval_user_id_dataset(
    dataset_path: str,
    model_path: str,
    seed: int = 0,
    train_batch_size: int = 128,
    eval_batch_size: int = 256,
    num_epochs: float = 1.0,
    num_train_steps: int = 6000,
    lr: float = 5e-5,
    num_proc: int = 24,
):
    set_seed(seed)
    ds_by_user: DatasetDict = datasets.load_from_disk(dataset_path)
    user_ids = list(ds_by_user.keys())
    print(f"Users ({len(user_ids)}):", ", ".join(user_ids[:10]), "..." if len(user_ids) > 10 else "")

    # map + tokenize per user
    print("Mapping labels...")
    ds_by_user = DatasetDict({
        uid: ds.map(apply_fixed_mapping, desc=f"map[{uid}]", num_proc=num_proc)
        for uid, ds in ds_by_user.items()
    })
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    print("Tokenizing & aligning...")
    ds_by_user = DatasetDict({
        uid: ds.map(lambda ex: tokenize_and_align_labels(ex, tokenizer),
                    batched=False, desc=f"tok[{uid}]", num_proc=num_proc)
        for uid, ds in ds_by_user.items()
    })

    # fixed config from your constants
    label2id = {v: k for k, v in ID2LABEL.items()}

    for test_uid in user_ids:
       
            print("\n" + "="*70)
            print(f"LOO: test user = {test_uid}")
            print("="*70)

            train_ds = _concat_except(ds_by_user, excluded_user=test_uid)
            train_ds = train_ds.shuffle(seed=seed)
            test_ds  = ds_by_user[test_uid]
            if len(test_ds) == 0:
                print(f"[skip] no examples for {test_uid}")
                continue
    

            config = AutoConfig.from_pretrained(
                model_path, num_labels=NUM_LABELS,
                id2label=ID2LABEL, label2id=label2id
            )
            model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
            collator = DataCollatorForTokenClassification(tokenizer)

            args = TrainingArguments(
                output_dir="./_tmp",  # required by HF; we won't save anything
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=eval_batch_size,
                learning_rate=lr,
                num_train_epochs=num_epochs,
                max_steps = 2000,
                logging_strategy="no",

                eval_strategy="no",
                # eval_strategy = "steps",
                # eval_steps = 2000,
                save_strategy="no",
                report_to=[],
                seed=seed,
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset = test_ds, 
                tokenizer=tokenizer,
                data_collator=collator,
                compute_metrics=compute_metrics,
            )

            trainer.train()

            preds = trainer.predict(test_ds)
            y_true, y_pred = _y_true_pred_from_logits(preds.predictions, preds.label_ids)

            target_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
            print(classification_report(
                y_true, y_pred,
                labels=list(range(NUM_LABELS)),
                target_names=target_names,
                digits=4, zero_division=0
            ))
            trainer.save_model(f"path/to/temp_checkpoints/user_{test_uid}")


if __name__ == "__main__":
    

    # Load dataset
    ds = datasets.load_from_disk("path/to/data/evkd_simulation_qwen_ds/")

    # Ensure validation split
    seed = 0
    if "train" not in ds:
        raise RuntimeError("split dataset into train test first")
    if "validation" not in ds and "test" in ds:
        ds = DatasetDict({"train": ds["train"], "validation": ds["test"]})
    elif "validation" not in ds:
        split = ds["train"].train_test_split(test_size=0.1, seed=seed)
        ds = DatasetDict({"train": split["train"], "validation": split["test"]})

    # Limit dataset for development/training
    train_num = 4_000_000
    if len(ds["train"])< train_num:
        train_num = 82300

    ds = DatasetDict({
        "train": ds["train"].select(range(train_num)),
        "validation": ds["validation"].select(range(1_000))
    })
    print(f"Validation set size: {len(ds['validation'])}")

    # Apply fixed mapping
    ds_mapped = ds.map(apply_fixed_mapping, desc="Apply fixed label mapping", num_proc=24)

    # ====== Model List and Training Configurations ======
    model_list = [
        # "roberta-base",
        # "roberta-large",
        "deberta-v3-base",
        # "deberta-v3-large",
        # "bert-large-uncased",
        # "bert-base-uncased"
    ]

    # Recommended per-model hyperparameters (based on model size)
    model_train_cfg = {
        "bert-base-uncased": {"lr": 3e-5, "steps": 3000},
        "roberta-base":      {"lr": 3e-5, "steps": 4500},  # slightly lower than BERT-base
        "roberta-large":     {"lr": 2e-5, "steps": 5000},  # large → conservative LR.
        "deberta-v3-base":   {"lr": 2e-5, "steps": 3000},  
        "deberta-v3-large":  {"lr": 1e-5, "steps": 3000},  
        "bert-large-uncased":{"lr": 2e-5, "steps": 3000},  
    }

    # ====== Loop through models ======
    for model in model_list:
        model_path = f"path/to/model_checkpoints/{model}/"
        output_dir = f"path/to/trained_checkpoints/multitask_evkd_{model}"

        cfg = model_train_cfg.get(model, {"lr": 5e-5, "steps": 3000})
        lr = cfg["lr"]
        steps = cfg["steps"]

        print("\n" + "="*80)
        print(f"Training model: {model}")
        print(f"Model path: {model_path}")
        print(f"Output dir: {output_dir}")
        print(f"LR: {lr} | Steps: {steps}")
        print("="*80)
        if "roberta" in model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        ds_tok = ds_mapped.map(
            lambda ex: tokenize_and_align_labels(ex, tokenizer),
            batched=False,
            desc=f"Tokenizing for {model}",
            num_proc=24
        )

        # Train model
        train_target_token_classifier(
            dataset=ds_tok,
            wandb_name = f"{model}_evkd_simulation", 
            model_name=model_path,
            tokenizer = tokenizer,
            output_dir=output_dir,
            seed=seed,
            train_batch_size=128,
            eval_batch_size=256,
            lr=lr,
            num_epochs=1,
            train_steps=steps,
            warmup_ratio=0.10, 
        )





    # LOO
    # dataset_path = "path/to/data/real_user_hf_dataset"  # built via build_user_id_datasets()
    # model_path = "path/to/model_checkpoints/bert-base-uncased"
    # loo_eval_user_id_dataset(
    #     dataset_path=dataset_path,
    #     model_path=model_path,
    #     seed=0,
    #     train_batch_size=64,
    #     eval_batch_size=128,
    #     num_epochs=1.0,
    #     lr=5e-5,
    #     num_proc=24,
    # )