from typing import Iterable, List, Set
import math
from wordfreq import word_frequency
import random 

def stratified_sample_by_frequency(
    words: Iterable[str],
    n_samples: int,
    n_bands: int = 5,
    lang: str = "en",
    rng: random.Random = None,
) -> Set[str]:
    """
    Stratified sampling of words across frequency bands.

    - Bins by equal-width intervals in log10 frequency space.
    - Samples round-robin across bands until n_samples are collected.
    - Deterministic if `rng` is provided (use your existing `rnd`).

    Returns a set of sampled words (size == n_samples unless not enough words).
    """
    words = list(words)
    if len(words) <= n_samples:
        return set(words)  # nothing to stratify

    # Compute frequencies (floor at tiny epsilon so log is defined)
    eps = 1e-12
    freqs: List[float] = []
    for w in words:
        f = word_frequency((w or "").strip(), lang)
        # word_frequency can return 0 for very rare/OOV words -> floor it
        freqs.append(max(float(f), eps))

    # Log10 transform to spread heavy-tailed distribution
    logf = [math.log10(f) for f in freqs]
    lo, hi = min(logf), max(logf)
    # Edge case: all same frequency -> single band
    if hi - lo < 1e-12:
        n_bands = 1

    # Build bands
    bands: List[List[int]] = [[] for _ in range(n_bands)]
    for idx, lf in enumerate(logf):
        if n_bands == 1:
            b = 0
        else:
            # Map to [0, n_bands-1]; include hi in the last bin
            pos = (lf - lo) / (hi - lo)
            b = min(n_bands - 1, int(pos * n_bands))
        bands[b].append(idx)

    # Shuffle indices within each band (but deterministically)
    rng = rng or random
    for b in bands:
        rng.shuffle(b)

    # Round-robin draw across bands to ensure diversity
    chosen_idx: List[int] = []
    band_ptr = [0] * n_bands
    while len(chosen_idx) < n_samples:
        progressed = False
        for bi in range(n_bands):
            if len(chosen_idx) >= n_samples:
                break
            ptr = band_ptr[bi]
            if ptr < len(bands[bi]):
                chosen_idx.append(bands[bi][ptr])
                band_ptr[bi] += 1
                progressed = True
        if not progressed:
            # All bands exhausted before hitting n_samples (shouldn't happen unless words < n_samples)
            break

    return {words[i] for i in chosen_idx}
