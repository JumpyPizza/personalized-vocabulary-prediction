import datasets 
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
from utils import SentenceTokenizer, SentenceTokenizerFast
from functools import partial
import os
import itertools
from multiprocessing import Pool, Manager
import multiprocessing as mp


    
def find_samples_with_words(dataset, word_list, output_file):
    # filter the dataset, get only samples which contains words from the word list

    # Load the vocabulary knowledge dataset
    word_occurrences = {}
    word_counts = {}

    # Convert word list to set for faster lookup
    word_set = set(word_list)

    # Process each sample in the dataset
    for idx, sample in enumerate(tqdm(dataset['train'], desc="Find samples with words from the word list")):
       
        text_tokens_list = sample['tokens']
        text_tokens = [token.lower() for token_list in text_tokens_list for token in token_list]
        # Split text into words and convert to set for faster lookup
        text_words = set(text_tokens)
    
        # Find matching words
        matching_words = word_set.intersection(text_words)
        
        if matching_words:
            # Update word counts
            for word in matching_words:
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1
                
                # If word exceeds 2000 occurrences, remove it from word_set
                if word_counts[word] > 2000:
                    word_set.remove(word)
                    continue         
                # Add sample index to word occurrences
                if idx not in word_occurrences:
                    word_occurrences[idx] = []
                word_occurrences[idx].append(word)
    print("a total of ", len(word_occurrences), " samples in the dataset contains words from the word list")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(word_occurrences, f)






def check_word_occurrences(word_list, output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        word_occurrences = json.load(f)
    word_to_samples = {}

    for sample_id, words in word_occurrences.items():
        for word in words:
            if word not in word_to_samples:
                word_to_samples[word] = []
            word_to_samples[word].append(int(sample_id))
    print(" a total of ", len(word_to_samples), " words in the list shows up in the dataset")
    max_samples = max(len(samples) for samples in word_to_samples.values())
    print("the most frequent word shows up in ", max_samples, " samples")  
    min_samples = min(len(samples) for samples in word_to_samples.values())
    print("the least frequent word shows up in ", min_samples, " samples")

    for word in word_list:
        if word not in word_to_samples:
            print(word)
    return word_to_samples


def get_word_sentences(word_to_samples, dataset, min_tokens=10, max_tokens=50):
    sample_to_vocab = defaultdict(set)
    for word, sample_ids in word_to_samples.items():
        for sample_id in sample_ids:
            sample_to_vocab[sample_id].add(word.lower())

    word_sentences = {word.lower(): [] for word in word_to_samples.keys()}
    sentence_word_dict = defaultdict(set)
    sentence_tokenizer = SentenceTokenizerFast()
    for sample_id, vocab_words in tqdm(sample_to_vocab.items(), total=len(sample_to_vocab),
                                       desc="Processing samples"):
        # Get the sample text and lowercase it once.
    
        sample = dataset['train'][sample_id]
        # Split the text into sentences.
        sample_sentences = sample['sentences']
        sample_tokens = sample['tokens']
        # For each sentence, compute the intersection with the vocab words known to be in this sample.
        for sentence, tokens in zip(sample_sentences, sample_tokens):
            
            # no duplicate sentences in the dataset
            if min_tokens <= len(tokens) <= max_tokens:
                tokens = [tok.lower() for tok in tokens]
                # Get only the words that are both in the sentence and in the vocabulary flagged for this sample.
                common_words = vocab_words.intersection(tokens)
                if common_words:
                    sentence_word_dict[sentence].update(common_words)
                # # Append the sentence for each matching word.
                # for word in common_words:
                #     word_sentences[word].append(sentence)
            else:
                continue

    sentence_word_dict = {k:list(v) for k,v in sentence_word_dict.items()}
    return sentence_word_dict



if __name__ == "__main__":
    ############### 1. tokenize and save the dataset ###############
    WORD_LIST_PATH = "path/to/data/word_list.txt"

    OUTPUT_DIR = "path/to/generated_outputs/coca/"

    with open(WORD_LIST_PATH, "r", encoding="utf-8") as f:
        word_list = [line.strip("\n") for line in f.readlines() if line.strip("\n")]
    
    # datasets.config.DOWNLOADED_DATASETS_PATH = Path(DATASET_PATH) 
    # dataset = load_dataset("allenai/dolma", "v1_6-sample")

    # batch_size = 50000
    # n_process = 24
    # sentence_tokenizer = SentenceTokenizerFast()
    # print("finish loading sentence tokenizer")
    # batch_tokenize = sentence_tokenizer.batch_process_text
    # dataset_processed = dataset.map(batch_tokenize, batched=True, batch_size=batch_size, num_proc=n_process)
    # dataset_processed.save_to_disk("path/to/data/processed_dataset/")
    
    ############# 2. get the sentences for each word #############
    # get the sentences for each word
    # Convert word_occurrences to word -> sample_ids mapping
    # dataset = datasets.load_from_disk("path/to/data/coca_dataset/")
    # word_occurrences_path = os.path.join(OUTPUT_DIR, "word_occurrences_lowered.json")
    # # find_samples_with_words(dataset, word_list, word_occurrences_path)

    # ############# 3. get sentences with min-max words and deduplicate sentences #############
    # word_to_samples = check_word_occurrences(word_list, word_occurrences_path)
    # sentence_word_dict = get_word_sentences(word_to_samples, dataset, min_tokens=10, max_tokens=100)
    # with open(os.path.join(OUTPUT_DIR, "sentence_word_dict_10_100.json"), "w", encoding="utf-8") as f:
    #     json.dump(sentence_word_dict, f)
    
    
    with open(os.path.join(OUTPUT_DIR, "sentence_word_dict_10_100.json"), "r", encoding="utf-8") as f:
        sentence_word_dict = json.load(f)

    print("finish loading")
    sentence_matching_words = {}
    with open(os.path.join(OUTPUT_DIR,"unique_idx_sentences.txt"), "w", encoding="utf-8") as f:
        for idx, (sentence, word_list) in enumerate(tqdm(sentence_word_dict.items(), total=len(sentence_word_dict), desc="Writing sentences")):
            sentence_matching_words[idx] = word_list 
            f.write(sentence.replace("\n", " "))
            f.write("\n")
    with open(os.path.join(OUTPUT_DIR,"sentence_matching_words.json"), "w", encoding="utf-8") as f:
        json.dump(sentence_matching_words, f)
    # # ############### check the first 100 sentences #############
    # with open(os.path.join(OUTPUT_DIR,"unique_idx_sentences.txt"), "r", encoding="utf-8") as f:
    #     sentences = f.readlines()
    # with open(os.path.join(OUTPUT_DIR,"sentence_matching_words.json"), "r", encoding="utf-8") as f:
    #     sentence_words = json.load(f)
    # print(len(sentences))
    # with open(os.path.join(OUTPUT_DIR,"first_100_sentences.md"), "w", encoding="utf-8") as f:
    #     f.write("# First 100 Sentences with Matching Words\n\n")
    #     for i in range(100):
    #         f.write(f"## Sentence {i+1}:\n")
    #         f.write(f"```text\n{sentences[i].strip()}\n```\n")
    #         f.write(f"\nMatching words: {', '.join(sentence_words[str(i)])}\n")
    #         f.write("\n---\n\n")
   