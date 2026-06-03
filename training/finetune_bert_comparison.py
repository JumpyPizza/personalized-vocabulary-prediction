"""
wrap finetune_bert to run comparison over set of params
"""


from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Union
from pathlib import Path
from dataclasses import dataclass, replace
import json
import random
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
import matplotlib.pyplot as plt

# ---- stratified bands helper (provided in your project) ----
from frequency_sampling import stratified_sample_by_frequency

# ---- fixed mapping from earlier ----
LABEL_MAP = {0: -100, "0": -100, "1": 0, "2": 0, "3": 0, "4": 1, "5": 1}
ID2LABEL = {0: "L_1_2_3", 1: "L_4_5"}
NUM_LABELS = 2
ZERO_VALUES = {0, "0"}


def read_user_jsonl(path: Union[str, Path]) -> Dict[str, List[Dict]]:
    """
    Each line: { "<user>": [ { "word": w, "tokens": [[...],...], "labels": [[...],...] }, ... ] }
    Returns: {user_id -> list[items]}
    """
    users = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Reading JSONL")):
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
    """Map a single sentence's word-level labels using LABEL_MAP; zeros masked to -100."""
    out = []
    for y in labels:
        if y in ZERO_VALUES:
            out.append(-100)
        else:
            key = str(y).strip()
            out.append(LABEL_MAP.get(key, -100))
    return out


def tokenize_and_align_labels(tokens: List[str], labels: List[int], tokenizer, max_length=256) -> Dict:
    """
    tokens: word-level tokens
    labels: word-level labels already mapped to {-100, 0, 1}
    Returns tokenized inputs with subword-aligned labels.
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


def _masked_metrics_and_report_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Dict[str, float], Dict]:
    """
    Compute macro accuracy/precision/recall/F1 on positions where label != -100,
    plus a structured classification report (sklearn output_dict).
    """
    y_pred_all = np.argmax(logits, axis=-1)

    y_true: List[int] = []
    y_pred: List[int] = []

    for p_row, l_row in zip(y_pred_all, labels):
        mask = l_row != -100
        if np.any(mask):
            y_true.extend(l_row[mask].tolist())
            y_pred.extend(p_row[mask].tolist())

    if not y_true:
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        report = {"warning": "No labeled tokens after masking."}
        return metrics, report

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[ID2LABEL[0], ID2LABEL[1]],
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    return metrics, report


def compute_metrics(eval_pred):
    """
    Trainer callback — returns macro metrics over masked positions.
    (Used if you call trainer.evaluate, but we mainly use trainer.predict + custom above.)
    """
    logits, labels = eval_pred
    metrics, _ = _masked_metrics_and_report_from_logits(logits, labels)
    return metrics


@dataclass
class FinetuneConfig:
    base_ckpt_path: str
    output_dir: str
    n_words_train: int = 30  # how many words as training data
    seed: int = 0
    train_batch_size: int = 16  # per device train batch size
    eval_batch_size: int = 32
    lr: float = 5e-5
    num_epochs: float = 1.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    max_length: int = 256

    # sampling + reporting
    sampling_strategy: str = "random"  # "random" | "stratified"
    n_bands: int = 5                   # for stratified
    lang: str = "en"                   # for stratified

    # one file per run with all users
    save_run_report: bool = True


def _build_user_datasets_for_words(
    items: List[Dict], train_words: set, test_words: set, tokenizer, max_length: int
) -> Tuple["datasets.Dataset", "datasets.Dataset"]:
    """
    From one user's items -> build sentence-level HF datasets for train (train_words) and test (test_words).
    Each item has: word, tokens: [[...], ...], labels: [[...], ...]
    """
    def collect_for(target_set: set, split: str) -> List[Dict]:
        exs: List[Dict] = []
        for it in items:
            w = it.get("word")
            if w not in target_set:
                continue
            for idx, (toks, labs) in enumerate(zip(it.get("tokens", []) or [], it.get("labels", []) or [])):
                if split == "train":
                    if idx > 5:  # number of sentence samples per label for training
                        break
                if split == "test":
                    if idx > 2:  # number of sentence samples per label for testing
                        break
                mapped = apply_fixed_mapping_word_level(labs)
                if all(l == -100 for l in mapped):
                    continue
                tok_out = tokenize_and_align_labels(toks, mapped, tokenizer, max_length)
                exs.append(tok_out)
        return exs

    from datasets import Dataset  # local import to keep signature tidy

    train_examples = collect_for(train_words, "train")
    test_examples = collect_for(test_words, "test")

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
      - sample N words for fine-tuning (random or stratified), rest for testing
      - load the SAME base checkpoint (cfg.base_ckpt_path)
      - record classification report BEFORE fine-tuning (on that user's test words)
      - fine-tune
      - record classification report AFTER fine-tuning (same test set)
      - do NOT save the fine-tuned checkpoint
      - save ONE run file with all users
    """
    set_seed(cfg.seed)
    users_data = read_user_jsonl(jsonl_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_ckpt_path, use_fast=True)

    results: Dict[str, Dict] = {"users": {}}
    all_f1_after, all_acc_after = [], []
    all_delta_f1, all_delta_prec, all_delta_rec = [], [], []

    for user, items in tqdm(users_data.items(), desc="Users"):
        unique_words = sorted({it.get("word") for it in items if it.get("word") is not None})
        if len(unique_words) < 2:
            raise RuntimeError("no words for training")   # need at least 1 train word + 1 test word

        # rnd = random.Random(cfg.seed + (user if isinstance(user, int) else hash(str(user)) % (2**16)))
        rnd = random.Random(cfg.seed)
        n_train = min(cfg.n_words_train, max(1, len(unique_words) - 1))

        # Sampling
        if cfg.sampling_strategy == "stratified":
            train_words = stratified_sample_by_frequency(
                unique_words, n_samples=n_train, n_bands=cfg.n_bands, lang=cfg.lang, rng=rnd
            )
        else:
            train_words = set(rnd.sample(unique_words, n_train))
        test_words = set(unique_words) - set(train_words)
        if not test_words:
            w_move = rnd.choice(list(train_words))
            train_words.remove(w_move)
            test_words.add(w_move)

        # Datasets
        train_ds, test_ds = _build_user_datasets_for_words(items, train_words, test_words, tokenizer, cfg.max_length)

        # Model
        config = AutoConfig.from_pretrained(
            cfg.base_ckpt_path,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id={v: k for k, v in ID2LABEL.items()},
        )
        model = AutoModelForTokenClassification.from_pretrained(cfg.base_ckpt_path, config=config)

        collator = DataCollatorForTokenClassification(tokenizer)
        out_dir_user = Path(cfg.output_dir) / f"user_{user}"
        out_dir_user.mkdir(parents=True, exist_ok=True)

        args = TrainingArguments(
            output_dir=str(out_dir_user),
            per_device_train_batch_size=cfg.train_batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            learning_rate=cfg.lr,
            num_train_epochs=cfg.num_epochs,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            eval_strategy="no",
            save_strategy="no",
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
            compute_metrics=compute_metrics,
        )

        # BEFORE
        preds_before = trainer.predict(test_ds)
        metrics_before, report_before = _masked_metrics_and_report_from_logits(
            preds_before.predictions, preds_before.label_ids
        )

        # TRAIN
        trainer.train()

        # AFTER
        preds_after = trainer.predict(test_ds)
        metrics_after, report_after = _masked_metrics_and_report_from_logits(
            preds_after.predictions, preds_after.label_ids
        )

        # Aggregates
        all_f1_after.append(metrics_after["f1"])
        all_acc_after.append(metrics_after["accuracy"])

        delta_f1 = metrics_after["f1"] - metrics_before["f1"]
        delta_prec = metrics_after["precision"] - metrics_before["precision"]
        delta_rec = metrics_after["recall"] - metrics_before["recall"]

        all_delta_f1.append(delta_f1)
        all_delta_prec.append(delta_prec)
        all_delta_rec.append(delta_rec)

        results["users"][user] = {
            "n_train_words": len(train_words),
            "n_test_words": len(test_words),
            "train_words": sorted(list(train_words)),
            # "test_words": sorted(list(test_words)),
            "sampling_strategy": cfg.sampling_strategy,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "delta": {"f1": float(delta_f1), "precision": float(delta_prec), "recall": float(delta_rec)},
            "classification_report_before": report_before,
            "classification_report_after": report_after,
        }

        print(f"User {user} | ΔF1={delta_f1:.4f}  ΔPrecision={delta_prec:.4f}  ΔRecall={delta_rec:.4f}")

    # Macro across users
    results["macro_f1_after"] = float(np.mean(all_f1_after)) if all_f1_after else 0.0
    results["macro_acc_after"] = float(np.mean(all_acc_after)) if all_acc_after else 0.0
    results["macro_f1_change"] = float(np.mean(all_delta_f1)) if all_delta_f1 else 0.0
    results["macro_precision_change"] = float(np.mean(all_delta_prec)) if all_delta_prec else 0.0
    results["macro_recall_change"] = float(np.mean(all_delta_rec)) if all_delta_rec else 0.0

    # Save one run file (all users)
    if cfg.save_run_report:
        runs_dir = Path(cfg.output_dir) / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_tag = f"{cfg.sampling_strategy}_n{cfg.n_words_train}"
        run_file = runs_dir / f"run_{run_tag}.json"
        payload = {
            "config": {
                **cfg.__dict__,
            },
            "summary": {
                "macro_f1_after": results["macro_f1_after"],
                "macro_acc_after": results["macro_acc_after"],
                "macro_f1_change": results["macro_f1_change"],
                "macro_precision_change": results["macro_precision_change"],
                "macro_recall_change": results["macro_recall_change"],
            },
            "users": results["users"],
        }
        with run_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[Saved run report] {run_file}")

    return results


def run_comparison(
    jsonl_path: Union[str, Path],
    strategy_list : List, 
    base_cfg: FinetuneConfig,
    n_words_list: Iterable[int] = (30, 50, 70, 100),
    output_name : str = "Summary"
):
    """
    Runs experiments for random vs stratified sampling over multiple n_words_train values.
    Produces:
      - summary JSON + CSV in output_dir/summary/
      - plots: ΔF1 / ΔPrecision / ΔRecall vs n_words_train for both strategies
      - run-level JSONs with all users (already written by finetune_per_user)
    """
    summary_dir = Path(base_cfg.output_dir) / output_name
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    per_setting_results = {}

    settings = [(strategy, n) for strategy in strategy_list for n in n_words_list]

    for strategy, n in tqdm(settings, desc="Run comparison", total=len(settings)):
        cfg = replace(base_cfg, n_words_train=n, sampling_strategy=strategy)
        print(f"\n=== Running {strategy} sampling with n_words_train={n} ===")
        print(f"====== lr is {cfg.lr} ========")
        results = finetune_per_user(jsonl_path, cfg)
        per_setting_results[(strategy, n)] = results


        rows.append({
            "strategy": strategy,
            "n_words_train": n,
            "macro_f1_after": results["macro_f1_after"],
            "macro_acc_after": results["macro_acc_after"],
            "macro_f1_change": results["macro_f1_change"],
            "macro_precision_change": results["macro_precision_change"],
            "macro_recall_change": results["macro_recall_change"],
        })

    # Save summary JSON & CSV
    summary_json_path = summary_dir / "summary_results.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": base_cfg.__dict__,
            "results": rows,
        }, f, indent=2)

    summary_csv_path = summary_dir / "summary_results.csv"
    with summary_csv_path.open("w", encoding="utf-8") as f:
        f.write("strategy,n_words_train,macro_f1_after,macro_acc_after,macro_f1_change,macro_precision_change,macro_recall_change\n")
        for r in rows:
            f.write(
                f"{r['strategy']},{r['n_words_train']},"
                f"{r['macro_f1_after']:.6f},{r['macro_acc_after']:.6f},"
                f"{r['macro_f1_change']:.6f},{r['macro_precision_change']:.6f},{r['macro_recall_change']:.6f}\n"
            )

    # # ---- Plots ----
    # def plot_metric(metric_key: str, ylabel: str, filename: str):
    #     xs = list(n_words_list)
    #     for strategy in ["random", "stratified"]:
    #         ys = [per_setting_results[(strategy, n)][metric_key] for n in n_words_list]
    #         plt.plot(xs, ys, marker="o", label=strategy)
    #     plt.xlabel("n_words_train")
    #     plt.ylabel(ylabel)
    #     plt.title(f"{ylabel} vs n_words_train")
    #     plt.grid(True, alpha=0.3)
    #     plt.legend()
    #     out_path = summary_dir / filename
    #     plt.savefig(out_path, bbox_inches="tight", dpi=180)
    #     plt.clf()

    # plot_metric("macro_f1_change", "ΔF1 (after - before)", "delta_f1_vs_n.png")
    # plot_metric("macro_precision_change", "ΔPrecision (after - before)", "delta_precision_vs_n.png")
    # plot_metric("macro_recall_change", "ΔRecall (after - before)", "delta_recall_vs_n.png")

    print(f"\nSaved summary to: {summary_json_path}")
    print(f"Saved CSV to:     {summary_csv_path}")
    print(f"Saved plots to:   {summary_dir / 'delta_f1_vs_n.png'}, "
          f"{summary_dir / 'delta_precision_vs_n.png'}, {summary_dir / 'delta_recall_vs_n.png'}")


if __name__ == "__main__":
    user_path = "path/to/data/sentence_level_real_user_labels.json"
    ckpt_path = "path/to/trained_checkpoints/multitask_bert-base"
    # ckpt_path = "path/to/trained_checkpoints/multitask_bert_fractions/bert_7pct_users/checkpoint-3000"
        
    cfg = FinetuneConfig(
        base_ckpt_path=ckpt_path,
        output_dir="./output",
        # lr = 1e-5, 
        n_words_train=30,   # run_comparison() will override per run
        train_batch_size=128,
        eval_batch_size=1024,
        num_epochs=1,
        sampling_strategy="random",  # run_comparison() will override per run
        n_bands=5, # only related to stratefied
        lang="en",
        save_run_report=True,
    )

    # Run both strategies across N ∈ {30, 50, 70, 100}, create reports + plots
    run_comparison(user_path, ["random"], cfg, 
                    # n_words_list=(30, 40, 120, 150, 200), # 30, 40,  
                    # n_words_list=(50, 60, 70, 80, 90, 100), # 30, 40,   
                    n_words_list=(20, 180, 200),
                    output_name = "summary_bert")
