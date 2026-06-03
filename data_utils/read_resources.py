import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union


def read_sentences_words(path: Union[str, Path]) -> Dict[int, List[str]]:
    """Load sentence-index to matching target-word mapping."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(idx): words for idx, words in data.items()}


def load_labels(path: Union[str, Path]) -> Dict:
    """
    Load token-label JSONL produced by TokenLabeler.save_labels.

    Returns:
      {
        "tokens": [...],
        "user_labels": {
          user_id: {"labels": [...], "true_labels": [...]}
        }
      }
    """
    labels = {"tokens": [], "user_labels": defaultdict(lambda: {"labels": [], "true_labels": []})}

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            labels["tokens"].append(sample["tokens"])
            for user_name, user_labels in sample.get("user_labels", {}).items():
                labels["user_labels"][user_name]["labels"].append(user_labels["labels"])
                labels["user_labels"][user_name]["true_labels"].append(user_labels["true_labels"])

    labels["user_labels"] = dict(labels["user_labels"])
    return labels
