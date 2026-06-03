# pipeline for building data needed for our experiments


from __future__ import annotations
from typing import Dict, List, Sequence, Set, Tuple
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
import json
import random
import os 
import warnings

# Pipeline 1. given a list of words, get contexts for each word 
def build_word_contexts_jsonl(
    dataset,                        # DatasetDict with "train" split; fields: "sentences" (List[str]), "tokens" (List[List[str]])
    word_list: Sequence[str],       # target words, should always be lower-cased
    output_jsonl_path: str | Path,  # where to write the JSONL (one line per word), containing the contexts for the word
    min_tokens: int = 10,           # keep sentences with token length in [min_tokens, max_tokens]
    max_tokens: int = 50,
    cap_per_word: int = 2000,       # max number of samples to scan/retain per word (speed)
    max_samples_per_word: int = 20, # max number of sentence contexts to write per word
    lowercase: bool = True,
    seed: int = 0,                  # shuffle candidate samples deterministically for fair coverage
) -> Dict[str, int]:
    """
    Write a JSONL where each line is:
      { "word": { "word": "<w>", "sentences": [s1, s2, ..], "tokens": [[s1 toks..], [s2 toks..], ...] } }

    Produce the following input required by UserLabelsApplier.
    Returns a small summary: { "words_written": <count> }
    """
    assert max_samples_per_word > 0 and cap_per_word > 0, "the sample numbers need to be positive"
    rnd = random.Random(seed)
    output_jsonl_path = Path(output_jsonl_path)
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) One scan to build word -> sample_ids 
    target_words: Set[str] = {w.lower() if lowercase else w for w in word_list}
    remaining_for_scan = set(target_words)  # words still under cap for scanning
    counts_scan = Counter()
    word_to_samples: Dict[str, List[int]] = defaultdict(list)

    for idx, sample in enumerate(tqdm(dataset["train"], desc="Scanning samples for target words")):
        if not remaining_for_scan:
            break  # all words reached the scan cap
        # Note that one sample contains several sentences 
        toks = (tok.lower() for sent in sample["tokens"] for tok in sent) if lowercase \
               else (tok for sent in sample["tokens"] for tok in sent)
        sample_word_set = set(toks)

        hits = sample_word_set.intersection(remaining_for_scan)
        if not hits:
            continue

        for w in list(hits):
            word_to_samples[w].append(idx)
            counts_scan[w] += 1
            if counts_scan[w] >= cap_per_word:
                remaining_for_scan.discard(w)

    # 2) Invert to sample -> candidate words (only samples that matter)
    sample_to_words: Dict[int, Set[str]] = defaultdict(set)
    for w, sids in word_to_samples.items():
        for sid in sids:
            sample_to_words[sid].add(w)

    # 3) Process only relevant samples to collect per-word contexts
    #    Fairness: shuffle the sample order so frequent words don't dominate early.
    candidate_sample_ids = list(sample_to_words.keys())
    rnd.shuffle(candidate_sample_ids)

    # Per-word accumulators (with per-word sentence dedupe)
    contexts_sentences: Dict[str, List[str]] = defaultdict(list)
    contexts_tokens: Dict[str, List[List[str]]] = defaultdict(list)
    seen_sentence_texts_per_word: Dict[str, Set[str]] = defaultdict(set)
    written_counts: Counter = Counter()

    # Track which words still need more contexts to early-exit
    words_needing_contexts: Set[str] = {w for w in target_words}

    for sid in tqdm(candidate_sample_ids, desc="Collecting contexts"):
        if not words_needing_contexts:
            break

        sample = dataset["train"][sid]
        candidate_words = sample_to_words[sid]

        for sent_text, sent_tokens in zip(sample["sentences"], sample["tokens"]):
            if not (min_tokens <= len(sent_tokens) <= max_tokens):
                continue

            toks_norm = [t.lower() for t in sent_tokens] if lowercase else list(sent_tokens)
            present = set(toks_norm).intersection(candidate_words)
            if not present:
                continue

            # Add this sentence for every present word that still needs contexts
            for w in list(present.intersection(words_needing_contexts)):
                if sent_text in seen_sentence_texts_per_word[w]:
                    continue
                contexts_sentences[w].append(sent_text)
                contexts_tokens[w].append(sent_tokens)
                seen_sentence_texts_per_word[w].add(sent_text)
                written_counts[w] += 1
                if written_counts[w] >= max_samples_per_word:
                    words_needing_contexts.discard(w)

    # 4) Write the JSONL (skip words with zero collected contexts)
    words_written = 0
    with output_jsonl_path.open("w", encoding="utf-8") as f:
        for w in sorted(contexts_sentences.keys()):
            if not contexts_sentences[w]:
                continue
            payload = {
                "word": {
                    "word": w,
                    "sentences": contexts_sentences[w],
                    "tokens": contexts_tokens[w],
                }
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            words_written += 1

    return {"words_written": words_written}



# Pipeline 2. Given the user label: word -> label, based on the contextual information, 
# label the sentence that contain the word
class UserLabelsApplier:
    """

    Output formats:
      - apply_from_loaded(..., output_jsonl_path): one JSONL where each line is:
          { "user_name": [ { "word": "<w>", "tokens": [[...], ...], "labels": [[...], ...] }, ... ] }
      - apply_from_loaded_to_dir(..., output_dir): writes one JSONL per user:
          <output_dir>/<user>.jsonl  with a single line:
          { "user_name": [ ... ] }

    Behavior:
      - Loads the base once via load_base(...). Subsequent apply calls reuse the cache.
      - Skips words with label 'NA'.
      - Drops sentences whose labels are all 0. Skips a word entry only if all its sentences were dropped.
    """

    def __init__(self,  case_insensitive: bool = True) -> None:
        self.case_insensitive = case_insensitive # should always keep this True and lower the words
        self._base_cache: Dict[str, Dict] | None = None   # word -> {"word", "sentences", "tokens"}

    # ---------- utilities ----------
    def _norm(self, s: str) -> str:
        # currently, norm only lowers a word 
        return s.lower() if self.case_insensitive else s

    @staticmethod
    def _username_from_filename(fname: str) -> str:
        stem = Path(fname).stem
        parts = stem.split("_")
        return parts[-1] if len(parts) > 1 else stem

    def _read_user_labels_dir(self, user_labels_dir: str | os.PathLike) -> Dict[str, Dict[str, str]]:
        """
        Return { user_name: { word: label, ... }, ... }
        Accepts both <word>\t<label> and "<word> <label...>" (space delimited).
        """
        result: Dict[str, Dict[str, str]] = {}
        for file in sorted(Path(user_labels_dir).glob("*.txt")):
            user = self._username_from_filename(file.name)
            user_map: Dict[str, str] = {}
            with file.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    if "\t" in line:
                        word, label = line.split("\t", 1)
                    else:
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        word, label = parts[0], " ".join(parts[1:])
                    user_map[self._norm(word)] = label
            result[user] = user_map
        return result

    def _labels_for_word(self, tokens_batch: List[List[str]], word: str, label_value: str, lebeling_schem : str) -> List[List]:
        """
        IMPORTANT: 
        Inheritating from 12k dataset, we use str(1), str(2), str(3) for not known, str(4), str(5) for known
        Create token-level labels for each sentence in tokens_batch:
        label_value when token==word ; else 0.
   
        """
        if lebeling_schem == "binary":
            map_table = {
                "1" : "5", 
                "0" : "1"
            }
        else:
            map_table = {
                "5" : "5", 
                "4" : "4", 
                "3" : "3", 
                "2" : "2", 
                "1" : "1", 
            }
        w = self._norm(word)
        out: List[List] = []
        for toks in tokens_batch:
            labels = []
            for t in toks:
                if self._norm(t) == w:
                    label = map_table[label_value]
                else:
                    label = 0
                labels.append(label)
            
            out.append(labels)
        return out

    @staticmethod
    def _filter_zero_label_sentences(
        tokens_batch: List[List[str]],
        labels_batch: List[List],
    ) -> Tuple[List[List[str]], List[List]]:
        """
        Drop sentences where all labels are 0. Keep alignment between tokens and labels.
        Returns (filtered_tokens_batch, filtered_labels_batch).
        """
        keep_tokens: List[List[str]] = []
        keep_labels: List[List] = []
        for toks, labs in zip(tokens_batch, labels_batch):
            if any(l != 0 for l in labs):
                keep_tokens.append(toks)
                keep_labels.append(labs)
        return keep_tokens, keep_labels

    # ---------- load base once ----------
    def load_base(self, base_jsonl_path: Union[str, os.PathLike, Iterable[Union[str, os.PathLike]]]) -> None:
        """
        Read and cache base JSONL file(s).

        - Accepts a single file path OR an iterable of file paths.
        - When multiple files are given, loads them in order and merges by word.
        - For each word, sentences are deduped by exact sentence text to avoid duplicates
        while preserving first-seen order; tokens remain aligned with kept sentences.

        Final cache shape:
        { word: {"word": w, "sentences": [str, ...], "tokens": [[str, ...], ...]} }
        """
        # Normalize input to a list of Paths, preserving given order
        if isinstance(base_jsonl_path, (str, os.PathLike)):
            files = [Path(base_jsonl_path)]
        else:
            files = [Path(p) for p in base_jsonl_path]

        cache: Dict[str, Dict] = {}
        # Track seen sentences per word to avoid dupes across files
        seen_sentences_per_word: Dict[str, set] = {}

        for file_path in files:
            with file_path.open("r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"loading contextual file: {file_path.name}"):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    # each line: { "word": { "word": "<w>", "sentences": [...], "tokens": [[...], ...] } }
                    (k, v), = obj.items()
                    w = self._norm(v["word"])
                    sents: List[str] = v.get("sentences", []) or []
                    toks_batch: List[List[str]] = v.get("tokens", []) or []

                    # shape validation
                    assert len(sents) == len(toks_batch)
                    
                    pairs = list(zip(sents, toks_batch))

                    if w not in cache:
                        cache[w] = {"word": w, "sentences": [], "tokens": []}
                        seen_sentences_per_word[w] = set()

                    # Merge while deduping by sentence text (per word)
                    seen = seen_sentences_per_word[w]
                    for s, t in pairs:
                        if s not in seen:
                            cache[w]["sentences"].append(s)
                            cache[w]["tokens"].append(t)
                            seen.add(s)
        print(f"loaded {len(cache)} words")
        self._base_cache = cache

    # ---------- build per-user entries using cached base ----------
    def _build_entries_for_user(self, user_word_labels: Dict[str, str], lebeling_schem, sample_num = 10, ) -> List[Dict]:
        """
        Using cached base, build the list of:
        { "word": w, "tokens": [[...], ...], "labels": [[...], ...] }
        for a single user.
        - Iterate over user's label words 
        - Error if a labeled word isn't in the base cache
        - Skip 'NA'
        - Compute labels
        - Drop sentences whose labels are all 0
        - Skip the word entirely only if no sentences remain after filtering
        """
        if self._base_cache is None:
            raise RuntimeError("Base not loaded. Call load_base(...) first.")

        entries: List[Dict] = []
        for w, value in user_word_labels.items():
            w_norm = w

            # Check that the word exists in base cache
            if w_norm not in self._base_cache:
                warnings.warn(f"Word '{w}' from user labels not found in base cache.", RuntimeWarning)
                continue
            if isinstance(value, str) and value.strip().lower() == "na":
                continue

            base_item = self._base_cache[w_norm]
            tokens_batch = base_item["tokens"]

            if not tokens_batch:
                continue

            # to speed up ================
            tokens_batch = tokens_batch[:100]
            # =============================

            labels_batch = self._labels_for_word(tokens_batch, w_norm, value, lebeling_schem)
            
            

            filt_tokens, filt_labels = self._filter_zero_label_sentences(tokens_batch, labels_batch)

            # shape valid
            for tok, lab in zip(filt_tokens, filt_labels):
                assert len(tok) == len(lab), "shape mismatch"

            if not filt_tokens:  # all sentences were all-zero → skip this word for this user
                continue
           
            entries.append({
                "word": w_norm,
                "tokens": filt_tokens[:sample_num],
                "labels": filt_labels[:sample_num],
            })
     
        return entries

    # ---------- multi-user apply using cached base ----------
    def apply_from_loaded(
        self,
        labeling_schema,
        user_labels_dir: str | os.PathLike,
        output_jsonl_path: str | os.PathLike,
    ) -> None:
        """
        Use the already-loaded base to create a single JSONL with one line per user:
          { "user_name": [ ...entries... ] }
          entry: {word, tokens[[], [], ...], labels[[], [], ...]}
        """
        if self._base_cache is None:
            raise RuntimeError("Base not loaded. Call load_base(...) first.")

        labels_by_user = self._read_user_labels_dir(user_labels_dir)

        out_path = Path(output_jsonl_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            for user, word_labels in tqdm(labels_by_user.items(), desc="building entries"):
                word_labels = {self._norm(k): v for k, v in word_labels.items()}
                entries = self._build_entries_for_user(word_labels, lebeling_schem = labeling_schema)
                if entries:
                    f.write(json.dumps({user: entries}, ensure_ascii=False) + "\n")

    def apply_from_loaded_to_dir(
        self,
        user_labels_dir: str | os.PathLike,
        output_dir: str | os.PathLike,
    ) -> None:
        """
        Same as apply_from_loaded, but writes one file per user in output_dir:
          output_dir/<user>.jsonl  (single-line JSONL)
        """
        if self._base_cache is None:
            raise RuntimeError("Base not loaded. Call load_base(...) first.")

        labels_by_user = self._read_user_labels_dir(user_labels_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for user, word_labels in labels_by_user.items():
            word_labels = {self._norm(k): v for k, v in word_labels.items()}
            entries = self._build_entries_for_user(word_labels)
            if not entries:
                continue
            with (out_dir / f"{user}.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({user: entries}, ensure_ascii=False) + "\n")


    
if __name__ == "__main__":
    import datasets

    DATASET_PATH = "path/to/data/coca_dataset_hf/"
    WORD_LIST_PATH = "path/to/data/word_list_12k.txt"
    
    evkd_user_labels = "path/to/data/evkd_user_labels"

    evkd_jsonl = "path/to/data/word_contexts_evkd.jsonl"
    w12k_jsonl = "path/to/data/word_contexts_12k.jsonl"


    # with open(WORD_LIST_PATH, "r", encoding="utf-8") as f:
    #     words_12k = [ln.strip() for ln in f if ln.strip()]

    # with open(user_labels, "r", encoding="utf-8") as f:
    #     user_labels = json.load(f)
    # words_evkd = user_labels["0"]["words"]

    # words = [w for w in words_evkd if w not in words_12k]
   
    # ds = datasets.load_from_disk(DATASET_PATH)

    
    # 1) build word contexts, needed for the first time with only words and raw text
    # stats = build_word_contexts_jsonl(
    #     ds,
    #     words,
    #     OUTPUT_JSONL,
    #     min_tokens=10,
    #     max_tokens=100,
    #     cap_per_word=2000,
    #     max_samples_per_word=20,
    #     lowercase=True,
    #     seed=0,
    # )
    # print("Summary:", stats)

    applier = UserLabelsApplier()
    # 2) Load big base once
    applier.load_base([evkd_jsonl, w12k_jsonl]) # 

    # 2) Write a single JSONL containing all users (one line per user)
    
    #### evkd users #######
    # applier.apply_from_loaded(
    #     user_labels_dir = evkd_user_labels,
    #     output_jsonl_path="path/to/data/sentence_level_evkd_user_labels.jsonl"
    # )


    # #### qwen_sim_data #####
    # qwen_sim_labels_with_user_id = "path/to/data/qwen_sim_labels/simulated_labels_id"
    # applier.apply_from_loaded(
    #     labeling_schema = "five-point",
    #     user_labels_dir = qwen_sim_labels_with_user_id,
    #     output_jsonl_path="path/to/data/qwen_sentence_user_labels_id.jsonl"
    # )

    ### rule-based data ### 
    # rule_labels = "path/to/data/random_labels/generation"
    # applier.apply_from_loaded(
    #     labeling_schema = "five-point",
    #     user_labels_dir = rule_labels,
    #     output_jsonl_path="path/to/data/random_labels/random_labels.jsonl"
    # )

    ### EVKD simulation ###
    labels = "path/to/data/evkd_simulation_qwen/"
    applier.apply_from_loaded(
        labeling_schema = "five-point",
        user_labels_dir = labels,
        output_jsonl_path="path/to/data/evkd_simulation_qwen.jsonl"
    )