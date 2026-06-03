
# @InProceedings{pmlr-v70-gal17a,
#   title = 	 {Deep {B}ayesian Active Learning with Image Data},
#   author =       {Yarin Gal and Riashat Islam and Zoubin Ghahramani},
#   booktitle = 	 {Proceedings of the 34th International Conference on Machine Learning},
#   pages = 	 {1183--1192},
#   year = 	 {2017},
#   editor = 	 {Precup, Doina and Teh, Yee Whye},
#   volume = 	 {70},
#   series = 	 {Proceedings of Machine Learning Research},
#   month = 	 {06--11 Aug},
#   publisher =    {PMLR},
#   url = 	 {https://proceedings.mlr.press/v70/gal17a.html},
# }

# Bayesian method 

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .base import ActiveLearningStrategy, StrategyConfig, register_strategy


@register_strategy("bald")
class BALD(ActiveLearningStrategy):
    """
    BALD with MC-Dropout
    """

    def select_words(self, model, dataset, select_k: int) -> List[str]:
        if self.cfg.mc_samples <= 1:
            raise ValueError("BALD requires cfg.mc_samples > 1 (e.g., 10).")

        device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        dl = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=lambda exs: _simple_collate(exs),
        )

        seq_scores: List[float] = []
        words: List[str] = []

        for pack in _mc_predict(model, dl, device, self.cfg.mc_samples):
            # pack: dict with:
            #   probs_mc: [S, B, T, C]
            #   labels:   [B, T]
            #   attn:     [B, T]
            #   spec:     [B, T]
            #   meta:     list[str]
            probs_mc = pack["probs_mc"]  # [S, B, T, C]
            labels  = pack["labels"]
            attn    = pack["attn"]
            spec    = pack["spec"]
            meta    = pack["meta"]

            S, B, T, C = probs_mc.shape

            # mean over MC samples: [B, T, C]
            probs_mean = probs_mc.mean(axis=0)

            # H(mean)
            ent_mean = _entropy_np(probs_mean)  # [B, T]

            # E[H]
            ent_each = _entropy_np(probs_mc)    # [S, B, T]
            ent_expect = ent_each.mean(axis=0)  # [B, T]

            # MI = H(mean) - E[H]
            mi_tokens = ent_mean - ent_expect   # [B, T]

            token_mask = (labels != -100) & (attn == 1) & (spec == 0)

            for b in range(B):
                valid = token_mask[b].astype(bool)
                vals = mi_tokens[b][valid]
                if vals.size == 0:
                    seq_score = 0.0
                else:
                    if self.cfg.seq_agg == "max":
                        seq_score = float(vals.max())
                    else:
                        seq_score = float(vals.mean())
                seq_scores.append(seq_score)
                words.append(meta[b])

        # reduce to per-word
        by_word: Dict[str, List[float]] = {}
        for s, w in zip(seq_scores, words):
            by_word.setdefault(w, []).append(s)
        word_scores = {
            w: (max(v) if self.cfg.word_agg == "max" else float(np.mean(v)))
            for w, v in by_word.items()
        }

        ranked = sorted(word_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [w for w, _ in ranked[:select_k]]


def _simple_collate(examples):
    """
    Stack already-tokenized fields. Padding:
      - input_ids/attn/spec with 0,
      - labels with -100.
    """
  
    keys = ["input_ids", "attention_mask", "labels", "special_tokens_mask"]
    batch = {k: torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(ex[k]) for ex in examples],
        batch_first=True,
        padding_value=0 if k != "labels" else -100
    ) for k in keys}
    batch["meta_word"] = [ex["meta_word"] for ex in examples]
    return batch


def _mc_predict(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    mc_samples: int
) -> Iterable[Dict[str, Any]]:
    """
    Yields dicts with stacked MC probabilities: probs_mc [S, B, T, C].
    """
    # enable dropout
    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    model.train()
    model.apply(enable_dropout)

    with torch.no_grad():
        for batch in dl:
            meta = batch.pop("meta_word")  # list[str]
            labels = batch["labels"].numpy()
            attn   = batch["attention_mask"].numpy()
            spec   = batch["special_tokens_mask"].numpy()
            batch  = {k: v.to(device) for k, v in batch.items()}

            probs_mc = []
            for _ in range(mc_samples):
                logits = model(**batch).logits  # [B, T, C]
                probs  = F.softmax(logits, dim=-1).cpu().numpy()
                probs_mc.append(probs)

            probs_mc = np.stack(probs_mc, axis=0)  # [S, B, T, C]
            yield {"probs_mc": probs_mc, "labels": labels, "attn": attn, "spec": spec, "meta": meta}


def _entropy_np(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Entropy along the last axis.
    Works with shapes [..., C] -> returns [...], keeps other dims.
    """
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)
