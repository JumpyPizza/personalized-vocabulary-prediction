"""
Load per-user JSONL → flatten user -> split tran_test by users -> get a mixed dataset
    Each example is one sentence:
      { "tokens": [...], "labels": [...], "word": "<target_word>" }
# note: strip label, as some user has label with space somehow, e.g. "5 "
"""


from __future__ import annotations
from typing import Dict, List, Iterable, Union, Tuple
from pathlib import Path
from itertools import chain
import json
import random
from tqdm import tqdm

import datasets
from datasets import Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets


def _iter_user_lines(paths: Iterable[Union[str, Path]]):
    """Yield payload_list from JSONL lines like: { "user_name": [ {...}, ... ] } or just [ {...}, ... ]."""
    for p in paths:
        with Path(p).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                (user, payload), = obj.items()
                yield user, payload
                # if isinstance(obj, list):           # no username wrapper
                #     yield obj
                # elif isinstance(obj, dict):         # { user: [...] }
                #     (user, payload), = obj.items()
                #     yield payload
                # else:
                #     raise RuntimeError("Incorrect JSONL line format.")


def _all_zero(seq) -> bool:
    """True if all labels in a sequence are 0/'0'."""
    return all((x == 0 or x == "0") for x in seq)


def _infer_label_value_dtype(examples: List[Dict]) -> str:
    """
    Inspect examples to choose Arrow dtype for label elements: 'int64' or 'string'.
    Falls back to 'string' if uncertain.
    """
    for ex in examples:
        labs = ex.get("labels")
        if not labs:
            continue
        for y in labs:
            if isinstance(y, int):
                return "int64"
            if isinstance(y, str):
                return "string"
    return "string"


def _to_columns(examples: List[Dict]) -> Dict[str, List]:
    """Columnize a list of sentence examples {tokens, labels, word} to dict-of-lists."""
    cols = {"tokens": [], "labels": [], "word": []}
    for ex in examples:
        cols["tokens"].append(ex["tokens"])
        cols["labels"].append(ex["labels"])
        cols["word"].append(ex.get("word", ""))  # keep for analysis
    return cols


def _build_dataset_chunked(
    examples: List[Dict],
    *,
    chunk_size: int = 50_000,
    label_dtype: str | None = None,
    desc: str = "Building HF Dataset (chunks)"
) -> Dataset:
    """
    Faster Dataset construction with progress:
    - predeclared features (no type guessing)
    - build in chunks, concatenate at the end
    """
    if not examples:
        return Dataset.from_dict({"tokens": [], "labels": [], "word": []})

    if label_dtype is None:
        label_dtype = _infer_label_value_dtype(examples)

    features = Features({
        "tokens": Sequence(Value("string")),
        "labels": Sequence(Value(label_dtype)),
        "word":   Value("string"),
    })

    chunks = []
    for start in tqdm(range(0, len(examples), chunk_size), desc=desc):
        end = min(start + chunk_size, len(examples))
        cols = _to_columns(examples[start:end])
        ds_chunk = Dataset.from_dict(cols, features=features)
        chunks.append(ds_chunk)

    return concatenate_datasets(chunks) if len(chunks) > 1 else chunks[0]


def build_user_split_datasets(
    jsonl_paths: Union[str, Path, List[Union[str, Path]]],
    test_size: float = 0.2,
    seed: int = 0,
    drop_all_zero_sentences: bool = True,
    *,
    chunk_size: int = 50_000,   # <-- new: control chunk size
) -> DatasetDict:
    """
    Load per-user JSONL → split by user → build big HF datasets with chunked progress.

    Each example is one sentence:
      { "tokens": [...], "labels": [...], "word": "<target_word>" }
    """
    # normalize input to list
    paths = [jsonl_paths] if isinstance(jsonl_paths, (str, Path)) else list(jsonl_paths)

    # 1) Load and flatten per user → sentence-level examples (but keep grouped by user for splitting)
    user_to_examples: Dict[int, List[Dict]] = {}
    for idx, items in tqdm(_iter_user_lines(paths), desc="Loading user data"):
        exs: List[Dict] = []
        # if idx == 400:
        #     break
        for item in items:
            word = item.get("word")
            tokens_batch = item.get("tokens", []) or []
            labels_batch = item.get("labels", []) or []
            for toks, labs in zip(tokens_batch, labels_batch):
                if drop_all_zero_sentences and _all_zero(labs):
                    continue
                labs = [str(lab) for lab in labs]
                exs.append({"tokens": toks, "labels": labs, "word": word})
        if exs:
            user_to_examples[idx] = exs

    if not user_to_examples:
        raise ValueError("No examples found in input JSONL(s).")

    # 2) Split by user (deterministic)
    print("Splitting train/test by user ...")
    users = sorted(user_to_examples.keys())
    rnd = random.Random(seed)
    rnd.shuffle(users)

    n_users = len(users)
    n_test = max(1, int(round(n_users * test_size)))
    test_users = set(users[:n_test])
    train_users = set(users[n_test:])
    if not train_users:
        train_users, test_users = set(users[n_test:]), set(users[:n_test])

    # 3) Collect examples for each split
    train_examples = list(chain.from_iterable(user_to_examples[u] for u in users if u in train_users))
    test_examples  = list(chain.from_iterable(user_to_examples[u] for u in users if u in test_users))

    if not train_examples:
        raise ValueError("Empty training set after split. Reduce test_size or check inputs.")
    if not test_examples:
        raise ValueError("Empty test set after split. Increase test_size or check inputs.")

    # 4) Build HF datasets (chunked + progress)
    print(f"Building train set ({len(train_examples):,} examples) ...")
    label_dtype = _infer_label_value_dtype(train_examples)  # use train to decide dtype
    train_ds = _build_dataset_chunked(
        train_examples,
        chunk_size=chunk_size,
        label_dtype=label_dtype,
        desc="Building train dataset (chunks)",
    )

    print(f"Building test set ({len(test_examples):,} examples) ...")
    test_ds = _build_dataset_chunked(
        test_examples,
        chunk_size=chunk_size,
        label_dtype=label_dtype,
        desc="Building test dataset (chunks)",
    )

    return DatasetDict({"train": train_ds, "test": test_ds})



######## 
# add a function to build dataset with user ids 
######## 

def build_user_id_datasets(
    jsonl_paths: Union[str, Path, List[Union[str, Path]]],
    drop_all_zero_sentences: bool = True,
    *,
    chunk_size: int = 50_000,
) -> DatasetDict:
    """
    Build a HuggingFace DatasetDict indexed by user ids (no train/test split).
    Each key is a user_id, and each value is that user's Dataset with columns:
      - tokens: Sequence[string]
      - labels: Sequence[int64 or string]  (dtype inferred)
      - word:   string
      - user_id: string  (constant per dataset, useful when concatenating later)

    Expected JSONL format per line: { "<user_id>": [ {word, tokens, labels}, ... ] }
    """
    # normalize input to list
    paths = [jsonl_paths] if isinstance(jsonl_paths, (str, Path)) else list(jsonl_paths)

    # ---- 1) Read & flatten per user (preserve user_id) ----
    user_to_examples: Dict[str, List[Dict]] = {}
    for user_id, items in tqdm(_iter_user_lines(paths), desc="Loading user data"):
     
        exs: List[Dict] = []
        for item in items:
            word = item.get("word")
            tokens_batch = item.get("tokens", []) or []
            labels_batch = item.get("labels", []) or []
            for toks, labs in zip(tokens_batch, labels_batch):
  
                if drop_all_zero_sentences and _all_zero(labs):
                    continue
                labs = [str(lab).strip(" ") for lab in labs]  # normalize to strings; we’ll infer dtype below
                exs.append({
                    "tokens": toks,
                    "labels": labs,
                    "word": word,
                    "user_id": str(user_id),
                })
        
        if exs:
            user_to_examples[str(user_id)] = exs

    if not user_to_examples:
        raise ValueError("No examples found in input JSONL(s).")

    # ---- 2) Infer label dtype globally to keep all per-user datasets compatible ----
    # (Look across users until we find a decisive dtype.)
    label_dtype = "string"
    for exs in user_to_examples.values():
        label_dtype = _infer_label_value_dtype(exs)
        if label_dtype != "string":
            raise RuntimeError("label type should be string")  # found int64

    # Predeclare common features
    features = Features({
        "tokens":  Sequence(Value("string")),
        "labels":  Sequence(Value(label_dtype)),
        "word":    Value("string"),
        "user_id": Value("string"),
    })

    # ---- 3) Build each user's dataset (chunked) ----
    user_ds: Dict[str, Dataset] = {}
    for uid, exs in tqdm(user_to_examples.items(), desc="Building per-user datasets"):
        if not exs:
            continue

        # chunked build to reduce memory
        chunks = []
        for start in range(0, len(exs), chunk_size):
            end = min(start + chunk_size, len(exs))
            cols = {
                "tokens":  [e["tokens"]  for e in exs[start:end]],
                "labels":  [e["labels"]  for e in exs[start:end]],
                "word":    [e.get("word", "") for e in exs[start:end]],
                "user_id": [uid] * (end - start),
            }
            ds_chunk = Dataset.from_dict(cols, features=features)
            chunks.append(ds_chunk)

        user_ds[uid] = concatenate_datasets(chunks) if len(chunks) > 1 else chunks[0]

    user_ids = sorted(user_to_examples.keys())
    seed = 0
    test_size = 0.05
    rnd = random.Random(seed)
    rnd.shuffle(user_ids)

    n_users = len(user_ids)
    n_test = max(1, int(round(n_users * test_size)))
    test_users = set(user_ids[:n_test])
    train_users = set(user_ids[n_test:])

 
    train_datasets = [user_ds[uid] for uid in train_users]
    test_datasets  = [user_ds[uid] for uid in test_users]

    train_ds = concatenate_datasets(train_datasets)
    test_ds  = concatenate_datasets(test_datasets)

    # ---- 6) Return combined DatasetDict ----
    full_ds = {"train": train_ds, "test": test_ds}

    return DatasetDict(full_ds)



if __name__ == "__main__":
    # this should be deprecated: dataset without user_id
    # ds = build_user_split_datasets("path/to/data/qwen_sentence_level_user_labels.json")
    # ds.save_to_disk("path/to/data/qwen_sent_level_hf_dataset")

    ##### sim dataset with user_id ### 
    # qwen_json_id = "path/to/data/qwen_sentence_user_labels_id.jsonl"
    # ds = build_user_id_datasets(qwen_json_id)
    # ds.save_to_disk("path/to/data/qwen_hf_dataset_id")

    ##### build real user dataset with id ###### 
    # ds_real_user = build_user_id_datasets("path/to/data/sentence_level_real_user_labels.json")
    # ds_real_user.save_to_disk("path/to/data/real_user_hf_dataset")

    #### rule based dataset #####
    # rule_json_id = "path/to/data/random_labels/random_labels.jsonl"
    # ds = build_user_id_datasets(rule_json_id)
    # ds.save_to_disk("path/to/data/rule_dataset_id")

    ### evkd simulation ###
    evkd_json = "path/to/data/evkd_simulation_qwen.jsonl"
    ds = build_user_id_datasets(evkd_json)
    ds.save_to_disk("path/to/data/evkd_simulation_qwen_ds")