
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Union
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import json
import random
import numpy as np

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
    tkn_path = "path/to/trained_checkpoints/full_finetune_bert"
    tokenizer = AutoTokenizer.from_pretrained(tkn_path, use_fast=True)

    results: Dict[str, Dict] = {"users": {}}
    all_f1, all_acc, all_mcc = [], [], []

    for user, items in tqdm(users_data.items(), desc="Users"):
        # if user == 0:
        #     continue
        # unique words in this user's data
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

        print(f"building dataset for user {user}")
        train_ds, test_ds = _build_user_datasets_for_words(items, train_words, test_words, tokenizer, cfg.max_length)

        # Fresh model from the SAME checkpoint for this user
        
        cfg.base_ckpt_path = f"path/to/temp_checkpoints/user_{user}"
   
        config = AutoConfig.from_pretrained(
            cfg.base_ckpt_path,
            # user_model_path,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id={v: k for k, v in ID2LABEL.items()},
        )
        print(f"loading model from model path: {cfg.base_ckpt_path}")
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
    return results

if __name__ == "__main__":
    user_path = "path/to/data/sentence_level_real_user_labels.json"
    # ckpt_path = "path/to/trained_checkpoints/full_finetune_bert"
    ckpt_path = None 
    cfg = FinetuneConfig(base_ckpt_path = ckpt_path, output_dir = "./output",
                        n_words_train = 70, train_batch_size = 70, eval_batch_size = 1024, num_epochs = 1)
    results = finetune_per_user(user_path, cfg)
    print(results)
    # f1 change over: 30 50 70 100 / random vs stratified
    

    # in-data-loo {'users': {0: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.45753854513168335, 'eval_accuracy': 0.7998640010879913, 'eval_precision': 0.7936858732568951, 'eval_recall': 0.7955009889420528, 'eval_f1': 0.794520040689574, 'eval_mcc': 0.5891840662669787, 'eval_runtime': 25.8516, 'eval_samples_per_second': 1385.951, 'eval_steps_per_second': 1.354, 'epoch': 1.0}}, 1: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.4397093951702118, 'eval_accuracy': 0.7946178335283394, 'eval_precision': 0.8041154573237956, 'eval_recall': 0.7664116919726072, 'eval_f1': 0.7751771059119438, 'eval_mcc': 0.5692799435801532, 'eval_runtime': 25.8809, 'eval_samples_per_second': 1384.378, 'eval_steps_per_second': 1.352, 'epoch': 1.0}}, 2: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.28491127490997314, 'eval_accuracy': 0.8671381936887922, 'eval_precision': 0.7780759250258298, 'eval_recall': 0.7611936977109508, 'eval_f1': 0.7691152230672802, 'eval_mcc': 0.5390053027638583, 'eval_runtime': 25.8068, 'eval_samples_per_second': 1388.353, 'eval_steps_per_second': 1.356, 'epoch': 1.0}}, 3: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.41200748085975647, 'eval_accuracy': 0.8198460408562958, 'eval_precision': 0.8111620824415124, 'eval_recall': 0.784420644044852, 'eval_f1': 0.7943699194505713, 'eval_mcc': 0.594982083395298, 'eval_runtime': 25.7912, 'eval_samples_per_second': 1389.196, 'eval_steps_per_second': 1.357, 'epoch': 1.0}}, 4: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.37159237265586853, 'eval_accuracy': 0.8373548053643808, 'eval_precision': 0.821459326864955, 'eval_recall': 0.8265296425825138, 'eval_f1': 0.8238231111471797, 'eval_mcc': 0.6479691323081038, 'eval_runtime': 25.8701, 'eval_samples_per_second': 1384.957, 'eval_steps_per_second': 1.353, 'epoch': 1.0}}, 5: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.2205915004014969, 'eval_accuracy': 0.905739934711643, 'eval_precision': 0.7382941231224889, 'eval_recall': 0.6243217131743689, 'eval_f1': 0.6572945575332501, 'eval_mcc': 0.34423906591769515, 'eval_runtime': 25.876, 'eval_samples_per_second': 1384.642, 'eval_steps_per_second': 1.353, 'epoch': 1.0}}, 6: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.19060373306274414, 'eval_accuracy': 0.9348949831320057, 'eval_precision': 0.615696316427002, 'eval_recall': 0.607688662747485, 'eval_f1': 0.611509418085885, 'eval_mcc': 0.223241408352785, 'eval_runtime': 25.8421, 'eval_samples_per_second': 1386.456, 'eval_steps_per_second': 1.354, 'epoch': 1.0}}, 7: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.2809304893016815, 'eval_accuracy': 0.8653391006284176, 'eval_precision': 0.6826413486975538, 'eval_recall': 0.6614604766606549, 'eval_f1': 0.6709590259731608, 'eval_mcc': 0.3434493221344419, 'eval_runtime': 25.8936, 'eval_samples_per_second': 1383.702, 'eval_steps_per_second': 1.352, 'epoch': 1.0}}, 8: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.4290216267108917, 'eval_accuracy': 0.7987378614367706, 'eval_precision': 0.7436337342511361, 'eval_recall': 0.7735496125583881, 'eval_f1': 0.7548424341231277, 'eval_mcc': 0.5163173969974344, 'eval_runtime': 25.7307, 'eval_samples_per_second': 1392.46, 'eval_steps_per_second': 1.36, 'epoch': 1.0}}, 9: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.4850486218929291, 'eval_accuracy': 0.7574743600206753, 'eval_precision': 0.7084588062969797, 'eval_recall': 0.6803774289808672, 'eval_f1': 0.6901761325365371, 'eval_mcc': 0.3878209046880779, 'eval_runtime': 25.7935, 'eval_samples_per_second': 1389.074, 'eval_steps_per_second': 1.357, 'epoch': 1.0}}, 10: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.42894110083580017, 'eval_accuracy': 0.8128672470076169, 'eval_precision': 0.7854736412579076, 'eval_recall': 0.7312541741042436, 'eval_f1': 0.7494779865074848, 'eval_mcc': 0.5138753589641308, 'eval_runtime': 25.8713, 'eval_samples_per_second': 1384.891, 'eval_steps_per_second': 1.353, 'epoch': 1.0}}, 11: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.4352194368839264, 'eval_accuracy': 0.8032755665587508, 'eval_precision': 0.7923274745831519, 'eval_recall': 0.797942080625939, 'eval_f1': 0.794696116936769, 'eval_mcc': 0.5902428517192921, 'eval_runtime': 24.7192, 'eval_samples_per_second': 1449.438, 'eval_steps_per_second': 1.416, 'epoch': 1.0}}, 12: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.544052004814148, 'eval_accuracy': 0.7261451419867262, 'eval_precision': 0.7283662397132544, 'eval_recall': 0.7252040721714066, 'eval_f1': 0.7248769921854767, 'eval_mcc': 0.45355928886925684, 'eval_runtime': 24.9142, 'eval_samples_per_second': 1438.097, 'eval_steps_per_second': 1.405, 'epoch': 1.0}}, 13: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.4521085023880005, 'eval_accuracy': 0.7809966813557477, 'eval_precision': 0.7897145742835365, 'eval_recall': 0.7798756410669672, 'eval_f1': 0.7788439207789803, 'eval_mcc': 0.56950523159686, 'eval_runtime': 24.9127, 'eval_samples_per_second': 1438.181, 'eval_steps_per_second': 1.405, 'epoch': 1.0}}, 14: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.3080032765865326, 'eval_accuracy': 0.8539756807486195, 'eval_precision': 0.7152676471222663, 'eval_recall': 0.743186657291417, 'eval_f1': 0.7275984139578775, 'eval_mcc': 0.4576034070093982, 'eval_runtime': 24.9569, 'eval_samples_per_second': 1435.632, 'eval_steps_per_second': 1.402, 'epoch': 1.0}}, 15: {'n_train_words': 50, 'n_test_words': 11946, 'metrics_after': {'eval_loss': 0.48227813839912415, 'eval_accuracy': 0.7755684909150256, 'eval_precision': 0.7851611165700012, 'eval_recall': 0.7853460825193115, 'eval_f1': 0.7755680244802131, 'eval_mcc': 0.5705071691051135, 'eval_runtime': 24.926, 'eval_samples_per_second': 1437.413, 'eval_steps_per_second': 1.404, 'epoch': 1.0}}}, 'macro_f1': 0.743303026460332, 'macro_acc': 0.8208647451892375, 'average_mcc': 0.49442387085430484}