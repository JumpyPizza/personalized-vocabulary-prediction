import numpy as np
from wordfreq import word_frequency
from typing import List, Dict, Optional
import json
from tqdm import tqdm

def assign_frequency_bands(
    words: List[str],
    n_bands: int = 20,
    method: str = 'quantile',
) -> np.ndarray:
    """
    Compute a frequency band (0..n_bands-1) for each word.
    - words: list of tokens
    - n_bands: how many bins
    - method: 'quantile' (equal-count) or 'equal_width' on [min_freq, max_freq]
    Returns:
      bands: an array of shape (len(words),) with integers in [0, n_bands-1].
    """
    # get raw frequencies
    freqs = np.array([word_frequency(w, 'en') for w in words], dtype=float)
    if method == 'quantile':
        edges = np.quantile(freqs, np.linspace(0, 1, n_bands+1))
    else:
        # equal width
        edges = np.linspace(freqs.min(), freqs.max(), n_bands+1)
    # np.digitize puts freq==edges[i] into bin i+1, so subtract 1
    bands = np.digitize(freqs, edges, right=False) - 1
    # clamp to valid range
    bands = np.clip(bands, 0, n_bands-1)
    return bands

def _generate_labels(
    words: List[str],
    bands: np.ndarray,
    know_probs: np.ndarray,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Generic label sampler:
    - know_probs: array of length n_bands, know_probs[i] = P(know|band=i)
    - returns dict word -> (0|1)
    """
    if seed is not None:
        np.random.seed(seed)
    # draw one U(0,1) per word
    draws = np.random.rand(len(words))
    labels = (draws < know_probs[bands]).astype(int)
    labels = [int(label) for label in labels]
    return dict(zip(words, labels))

def generate_low(
    words: List[str],
    n_bands: int = 20,
    know_min: float = 0.05,
    know_max: float = 0.20,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    LOW-level learner: knows very few words, even in the top bands.
    By default P(know) ramps linearly from 5% in the rarest band up to 20% in the most frequent.
    """
    bands = assign_frequency_bands(words, n_bands=n_bands, method='quantile')
    know_probs = np.linspace(know_min, know_max, n_bands)
    return _generate_labels(words, bands, know_probs, seed=seed)

def generate_intermediate(
    words: List[str],
    n_bands: int = 20,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    INTERMEDIATE-level learner: 
    - rarely knows the very low-freq words,
    - moderately knows mid-freq words,
    - almost certainly knows the top-freq words.
    
    Here we use a piecewise scheme:
      bands  0-4  (5/20) → 20% chance
      bands  5-14 (10/20) → 60% chance
      bands 15-19 (5/20) → 95% chance
    """
    bands = assign_frequency_bands(words, n_bands=n_bands, method='quantile')
    know_probs = np.zeros(n_bands, dtype=float)
    know_probs[:5] = 0.20
    know_probs[5:15] = 0.60
    know_probs[15:] = 0.95
    return _generate_labels(words, bands, know_probs, seed=seed)

def generate_high(
    words: List[str],
    n_bands: int = 20,
    know_min: float = 0.80,
    know_max: float = 0.99,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    HIGH-level learner: knows most words across the board, 
    but still slightly more likely to know high-freq ones.
    We ramp from 80% up to 99% over the bands.
    """
    bands = assign_frequency_bands(words, n_bands=n_bands, method='quantile')
    know_probs = np.linspace(know_min, know_max, n_bands)
    return _generate_labels(words, bands, know_probs, seed=seed)


if __name__ == "__main__":
    # word_file = "path/to/data/word_list.txt"
    # with open(word_file) as f:
    #     words = [w.strip() for w in f if w.strip()]
    # for i in tqdm(range(200)):
    #     low_labels   = generate_low(words, seed=None)
    #     mid_labels   = generate_intermediate(words, seed=None)
    #     high_labels  = generate_high(words, seed=None)
    #     # save labels to file

    #     with open(f"./output/random_low_labels_{i}.json", "w") as f:
    #         json.dump(low_labels, f)
    #     with open(f"./output/random_mid_labels_{i}.json", "w") as f:
    #         json.dump(mid_labels, f)
    #     with open(f"./output/random_high_labels_{i}.json", "w") as f:
    #         json.dump(high_labels, f)


    ######## write outputs ###########
    import os
    files = os.listdir("./output/")
    output_dir = "path/to/data/random_vocab_knowledge/"
    for file in files:
        file_name = file.split(".")[0]
        with open(f"./output/{file}", "r") as f:
            labels = json.load(f)
        output_fn = "".join(file_name.split("_"))
       
        with open(f"{output_dir}/random_{output_fn}.txt", "w") as f:
            for word, label in labels.items():
                if label == 1:
                    label_to_write = 5
                elif label == 0:
                    label_to_write = 1
                else:
                    raise ValueError(f"Invalid label: {label}")
                f.write(f"{word}\t{label_to_write}\n")

