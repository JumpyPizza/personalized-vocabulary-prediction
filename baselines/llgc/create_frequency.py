from wordfreq import zipf_frequency
import sys
import json 

word_list = []
with open("evkd_word_list.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        if line.strip():
            word_list.append(line.strip())

with open("vkd_word_list.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        if line.strip():
            if line not in word_list:
                word_list.append(line.strip())

##### word freq #####
# word_freqs = {w: zipf_frequency(w, "en") for w in word_list}

# # Sort words by frequency (descending)
# sorted_words = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)

# # Assign rank (1 = most frequent)
# ranked = [(word, rank + 1) for rank, (word, _) in enumerate(sorted_words)]

# # Write to file
# with open("wordfreq_rank.txt", "w", encoding="utf-8") as f:
#     for word, rank in ranked:
#         f.write(f"{word}\t{rank}\n")

##### corpora freq ######

with open("slimpajama_val_wordfreq.json", "r", encoding="utf-8") as f:
    word_counts = json.load(f)  # {"word1": count1, "word2": count2, ...}



for w in word_list:
    if w not in word_counts:
        word_counts[w] = 0


# Keep only the words of interest
subset_counts = {w: word_counts[w] for w in word_list}

# Sort by count (highest first), then assign ranks sequentially
sorted_words = sorted(subset_counts.items(), key=lambda x: x[1], reverse=True)

# Write ranks only for words in the given list
with open("corpus_rank.txt", "w", encoding="utf-8") as f:
    for rank, (word, _) in enumerate(sorted_words, start=1):
        f.write(f"{word}\t{rank}\n")