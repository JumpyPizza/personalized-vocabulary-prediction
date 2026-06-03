# active_learning/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from tqdm import tqdm

@dataclass
class StrategyConfig:
    """config for selection strategy"""
    batch_size: int = 64
    max_length: int = 1024
    device: str = "cuda"
    seq_agg: str = "mean"     # one example might contain multiple target words -> word score: "mean" | "max"
    word_agg: str = "mean"    # for one word, we might use several seqs, reduce to one score -> "mean" | "max"
    mc_samples: int = 0       # >0 enables MC-dropout averaging (BALD)


class ActiveLearningStrategy(ABC):
    """
    select words, given:
      - a model
      - an unlabeled test Dataset where each example contains a 'meta_word' field 
    Concrete strategies implement `select_words(...)`.
    """

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    @abstractmethod
    def select_words(
        self,
        model: torch.nn.Module,
        tokenizer,
        dataset,     # HF Datasets: meta_word, input_ids, attention_mask, labels, special_tokens_mask
        select_k: int
    ) -> List[str]:
        """Return up to `select_k` word strings to acquire next."""
        ...


    def _move_to(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in batch.items()}

    def _enable_dropout(self, m: torch.nn.Module):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    def _predict_token_probs(
        self,
        model: torch.nn.Module,
        dl,                         # torch DataLoader yielding dict batches incl. labels masks
        device: torch.device,
        mc_samples: int = 0
    ) -> Iterable[Dict[str, Any]]:
        """
        Yields per-batch dicts with:
          - 'probs':   float32 [B, T, C]
          - 'labels':  int64    [B, T]
          - 'attn':    int64    [B, T]
          - 'spec':    int64    [B, T]
          - 'meta':    list[str] length B  (words)
        """
        if mc_samples > 0:
            model.train()
            model.apply(self._enable_dropout)
        else:
            model.eval()

        with torch.no_grad():
            for batch in tqdm(dl, desc="predicting token probs"):
                meta = batch.pop("meta_word")                      # list[str]
                labels = batch["labels"]
                attn = batch["attention_mask"]
                spec = batch["special_tokens_mask"]
                batch = self._move_to(batch, device)
                if mc_samples <= 1:
                    logits = model(**batch).logits                 # [B, T, C]
                    probs = F.softmax(logits, dim=-1)
                else:
                    probs_accum = None
                    for _ in range(mc_samples):
                        logits = model(**batch).logits
                        p = F.softmax(logits, dim=-1)
                        probs_accum = p if probs_accum is None else probs_accum + p
                    probs = probs_accum / mc_samples

                yield {
                    "probs": probs.cpu().numpy(),
                    "labels": labels.numpy(),
                    "attn": attn.numpy(),
                    "spec": spec.numpy(),
                    "meta": meta,
                }

    @staticmethod
    def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        p = np.clip(p, eps, 1.0)
        return -np.sum(p * np.log(p), axis=-1)   # [T]

    @staticmethod
    def _aggregate(vals: np.ndarray, mode: str) -> float:
        if vals.size == 0:
            return 0.0
        if mode == "max":
            return float(vals.max())
        return float(vals.mean())

    def _sequence_scores_from_batches(
        self,
        batch_stream: Iterable[Dict[str, Any]]
    ) -> Tuple[List[float], List[str]]:
        """
        Convert the batch stream to per-example scores and aligned meta_word list.
        Sequence score = aggregate over token entropies on **target tokens**:
            mask = (labels != -100)  (safer for your data)
        """
        seq_scores: List[float] = []
        words: List[str] = []

        for pack in batch_stream:
            probs = pack["probs"]       # [B, T, C]
            labels = pack["labels"]     # [B, T]
            attn = pack["attn"]         # [B, T]
            spec = pack["spec"]         # [B, T]
            meta = pack["meta"]         # length B

            # token mask: keep tokens that are (labels != -100) AND attention_mask==1 AND not special
            token_mask = (labels != -100) & (attn == 1) & (spec == 0)

            B = probs.shape[0]
            for b in range(B):
                tok_ent = self._entropy(probs[b])                  # [T]
                valid = token_mask[b].astype(bool)
                seq_score = self._aggregate(tok_ent[valid], mode=self.cfg.seq_agg)
                seq_scores.append(seq_score)
                words.append(meta[b])

        return seq_scores, words

    def _reduce_to_word_scores(
        self,
        seq_scores: List[float],
        words: List[str]
    ) -> Dict[str, float]:
        """
        Aggregate multiple example scores per word -> a single word score.
        """
        by_word: Dict[str, List[float]] = {}
        for s, w in zip(seq_scores, words):
            by_word.setdefault(w, []).append(s)
        return {w: self._aggregate(np.array(v, dtype=float), self.cfg.word_agg) for w, v in by_word.items()}


# ---------- 
# simple registry to instantiate by name 
# ----------
_REGISTRY: Dict[str, Callable[..., ActiveLearningStrategy]] = {}

def register_strategy(name: str):
    def _wrap(cls):
        _REGISTRY[name] = cls
        return cls
    return _wrap

def use_strategy(name: str, *args, **kwargs) -> ActiveLearningStrategy:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown AL strategy '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](*args, **kwargs)
