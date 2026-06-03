
# uncertainty: entropy 

from __future__ import annotations
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import ActiveLearningStrategy, register_strategy


@register_strategy("uncertainty")
class UncertaintyEntropy(ActiveLearningStrategy):
    """
    Select words with highest average token entropy (over target tokens).
    """

    def select_words(self, model, dataset, select_k: int) -> List[str]:
        device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        # We assume dataset returns dicts including 'meta_word'
        dl = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=lambda examples: _simple_collate(examples),
        )
        batch_stream = self._predict_token_probs(model, dl, device, mc_samples=0)
        seq_scores, words = self._sequence_scores_from_batches(batch_stream)
        word_scores = self._reduce_to_word_scores(seq_scores, words)

        # rank by score desc
        ranked = sorted(word_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [w for w, _ in ranked[:select_k]]


def _simple_collate(examples):
    """
    examples: list of dicts from the HF Dataset; each has:
      - input_ids, attention_mask, labels, special_tokens_mask, meta_word
    We just stack tensors for collation
    """

    keys = ["input_ids", "attention_mask", "labels", "special_tokens_mask"]
    batch = {k: torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(ex[k]) for ex in examples],
        batch_first=True,
        padding_value=0 if k != "labels" else -100
    ) for k in keys}
    batch["meta_word"] = [ex["meta_word"] for ex in examples]
    return batch
