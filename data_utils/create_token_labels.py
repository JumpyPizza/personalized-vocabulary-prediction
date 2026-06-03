"""
Please note that this is deprecated: we do not use token-level split anymore, and use sentence-level split
create binary labels for words based on its context and learner labels
inputs: 
    - sentence_matching_words.json
    - usern.txt: word\tlabel
"""
import os
from collections import defaultdict
from tqdm import tqdm
import random
import json
from read_resources import load_labels, read_sentences_words
from utils import SentenceTokenizerFast
from copy import deepcopy


class TokenLabeler:
    def __init__(self, sentence_words, sentences, use_rule_for_tokenization=True):
        """
        based on the user-word-labels, create token level labels;
        since the labels are word-level, we can not really split the train-test set based on the sentences
            - there exists common words that connects (co-occurs) with basically all the words, so all the sentences end up in the same set
            - one method is to allow common words in both sets, and split sentences based on clusters in co-occurrence graph
            - currently we use the word level split: the words in the test set are labeled as -100 in training and do not contribute to the loss
        Args:
            sentence_words: Dict mapping sentence indices to target words it contains
            sentences: List of sentences corresponding to the sentence indices in sentence_words
        """
        random.seed(0)  # Keep same seed as word split for consistency
        self.all_samples = []
        self.original_labels = {}
        self.word_list = list(set([word for words in sentence_words.values() for word in words]))
        self.final_labels = {
            "tokens": [],
            "user_labels": {}
        }
        sentence_tokenizer = SentenceTokenizerFast()
        # Process each sentence
        max_length = 0
        min_length = 100
        for sentence_idx, words_in_sentence in tqdm(sentence_words.items(), desc="processing sentences"):
            # Get the sentence text and tokens
            sentence = sentences[int(sentence_idx)]
            tokens = sentence_tokenizer.tokenize_sentence(sentence, use_rule=use_rule_for_tokenization)
            tokens_length = len(tokens)
            if tokens_length > max_length:
                max_length = tokens_length
            if tokens_length < min_length:
                min_length = tokens_length
            if tokens_length<10 or tokens_length>50: # only keep sentences with 10-50 tokens; this should already be handled previously
                continue
            else:
                # Store tokens and words for this sentence
                self.all_samples.append({
                    "tokens": tokens,
                    "words_in_sentence": words_in_sentence,
                })
        print(f"max token length: {max_length}, min token length: {min_length}")
        self.train_test_split()
        # Randomly shuffle all samples
        random.shuffle(self.all_samples)
        print(f"done initializing tokens and words, total sample length: {len(self.all_samples)}")
    
    def train_test_split(self, train_ratio=0.8):
        """
        split train-test based on unique words
        """
        # random split the word list into train and test
        # random.seed(0)
        random.shuffle(self.word_list)
        train_size = int(len(self.word_list) * train_ratio)
        self.train_words = set(self.word_list[:train_size])
        self.test_words = set(self.word_list[train_size:])
        print(f"Total train words: {len(self.train_words)}, total test words: {len(self.test_words)}")
        

    def create_labels(self):
        raise NotImplementedError("create_labels is should be implemented in subclasses")


    def save_labels(self, save_labels_path):
        print("saving labels...")
        with open(save_labels_path, "w", encoding="utf-8") as f:
            # Save each sample as a separate JSON line
            for i in tqdm(range(len(self.final_labels["tokens"])), desc="saving labels"):
                sample = {
                    "tokens": self.final_labels["tokens"][i],
                    "user_labels": {}
                }
                for user_name in self.final_labels["user_labels"]:
                    sample["user_labels"][user_name] = {
                        "labels": self.final_labels["user_labels"][user_name]["labels"][i],
                        "true_labels": self.final_labels["user_labels"][user_name]["true_labels"][i]
                    }
                f.write(json.dumps(sample) + "\n")

class TokenLabelerWordLevel(TokenLabeler):
    def __init__(self, sentence_words, sentences):
        super().__init__(sentence_words, sentences)

    def create_labels(self, original_labels_dir):
        """Create labels for all tokens based on user labels.
            the structure for user labels is
                user_labels: {
                    user_name_1: {
                        labels: a list of labels, each index corresponds to each token in the sentence; test set tokens are labeled as -1
                        true_labels: test set tokens are labeled using their original labels
                        }
                    user_name_n...
                }
                   
        Args:
            original_labels_dir: directory containing the user-word-labels, each file is a user's labels
        """
        print("loading original labels...")
        assert self.train_words is not None, "train_words is not set"
        for file in os.listdir(original_labels_dir):

            if file.endswith(".txt"):
                # beginning processing for each user 
                user_name = file.split(".")[0].split("_")[-1]
                self.original_labels[user_name] = {}
                with open(os.path.join(original_labels_dir, file), "r", encoding="utf-8") as f:
                    # read all the labels for the current user
                    labels = [l for l in f.readlines() if l.strip("\n") != ""]
                    for line in labels:
                        word = line.strip("\n").split("\t")[0].lower()
                        label = line.strip("\n").split("\t")[1]
                       
                        self.original_labels[user_name][word] = label
                
                self.final_labels["user_labels"][user_name] = {
                    "labels": [],
                    "true_labels": []
                }

        # now all the users' labels are loaded
        # should already handle train-test split here
        # words not in train set should be labeled as 0
        print("processing labels...")
        self.token_label_count = defaultdict(lambda: 0) # count how many training samples a token has
        cleaned_samples_cnt = 0
        for sample in tqdm(self.all_samples, desc="creating labels"):
            tokens = sample["tokens"]
            words_in_sentence = sample["words_in_sentence"]
            
            # add user's labels for the current tokens
            temp_user_labels = {user_name: {
                "labels": [],
                "true_labels": []
            } for user_name in self.original_labels.keys()}
            for token in tokens:
                if token in words_in_sentence: # the token appears in the word_list
                    if token in self.train_words: # train set   
                        self.token_label_count[token] += 1
                        if self.token_label_count[token] < 2000: # limit the number of training samples for each token
                            for user_name, labels in self.original_labels.items():
                                temp_user_labels[user_name]["labels"].append(labels[token])
                                temp_user_labels[user_name]["true_labels"].append(labels[token])
                        else:
                            for user_name, labels in self.original_labels.items():
                                temp_user_labels[user_name]["labels"].append(0) 
                                temp_user_labels[user_name]["true_labels"].append(0)
                    else: # test set
                        for user_name, labels in self.original_labels.items():
                            temp_user_labels[user_name]["labels"].append(-1) # test set is labeled as -1
                            temp_user_labels[user_name]["true_labels"].append(labels[token])

                else: # the token was not labeled by the user (not in the word_list)
                    for user_name, labels in self.original_labels.items():
                        temp_user_labels[user_name]["labels"].append(0) # non shown words are labeled as 0
                        temp_user_labels[user_name]["true_labels"].append(0)
            # clean samples that only contains 0 
            
            first_labels = list(temp_user_labels.values())[0]["labels"]
            if all(label == 0 for label in first_labels):
                cleaned_samples_cnt += 1
                continue
            else:
                for user_name, labels in temp_user_labels.items():
                    self.final_labels["user_labels"][user_name]["labels"].append(labels["labels"])
                    self.final_labels["user_labels"][user_name]["true_labels"].append(labels["true_labels"])
                self.final_labels["tokens"].append(tokens)
        print(f"cleaned {cleaned_samples_cnt} samples")
        # ckeck that number of tokens is the same as the number of labels
        number_of_tokens = len(self.final_labels["tokens"])
        number_of_labels = len(self.final_labels["user_labels"]["user0"]["labels"])
        assert number_of_tokens == number_of_labels, "number of tokens is not the same as the number of labels"




"""
Token Labeling: 
  - for each word, randomly sample a small amount of sentences; 
  - In training set, only training words are labeled, others are labeled as 0;
      - To avoid introducing too many common words, starting from the least common words while tracking the occurence of other words;
  - In test set, only test words are labeled; To make it simple, in one sample we only consider one test word;
  - the prediction for a test label is a major vote of all the sentences for this test word;
"""

class TokenLabelerTrainTestSplit(TokenLabeler):
    """
    For each word, only a small amount of sentences are considered;
    Training and test set are split: 
        TokenLabelerWordLevel treats the task as sequence labeling and calculates test performance on all samples, which is unnecessary;
        In a context, we only care about the test token's labels, so a small amount of sentences to provide labels are enough.
        
    """
    def __init__(self, sentence_words, sentences, use_rule_for_tokenization):
        super().__init__(sentence_words, sentences, use_rule_for_tokenization)

    def create_labels(self, original_labels_dir, max_samples_per_word=20, test_k = 10): 
        """
        test_k: max number of test samples for each word; avoid too many samples in test set due to common words
        """
        assert self.train_words is not None, "train_words is not set"
        print("loading original labels...")
        for file in os.listdir(original_labels_dir):

            if file.endswith(".txt"):
                # beginning processing for each user 
                user_name = file.split(".")[0].split("_")[-1]
                self.original_labels[user_name] = {}
                with open(os.path.join(original_labels_dir, file), "r", encoding="utf-8") as f:
                    # read all the labels for the current user
                    labels = [l for l in f.readlines() if l.strip("\n") != ""]
                    for line in labels:
                        word = line.strip("\n").split("\t")[0].lower()
                        label = line.strip("\n").split("\t")[1]
                        
                        self.original_labels[user_name][word] = label

        word_sample_count = defaultdict(int)
        
        # New list to store filtered samples
        filtered_samples = []
        
        # Process samples in order
        for sample in self.all_samples:
            # Get words that are still under the max sample limit
            valid_words = [word for word in sample["words_in_sentence"] 
                         if word_sample_count[word] < max_samples_per_word]
            
            # Skip if no valid words remain
            if not valid_words:
                continue
                
            # Add sample since it contains valid words
            filtered_samples.append(sample)
            
            # Update counts for all words in the sample
            for word in valid_words:
                word_sample_count[word] += 1           
                        
        print(f"Filtered from {len(self.all_samples)} to {len(filtered_samples)} samples")
         
        test_samples = []
        test_sample_word_count = defaultdict(lambda: 0)
        for sample in filtered_samples:
            words_in_sentence = sample["words_in_sentence"]
            for word in words_in_sentence:
                if word in self.test_words:
                    if test_sample_word_count[word] < test_k:
                        test_samples.append(sample)
                        test_sample_word_count[word] += 1
                        break
        print(f"actual test words number: {len(test_sample_word_count)}")
        # start creating labels
        self.train_labels = self.create_labels_from_samples(filtered_samples)
        self.test_labels = self.create_labels_from_samples(test_samples)

        
    def create_labels_from_samples(self, samples):
        
        labels_for_users = deepcopy(self.final_labels)
        for user_name in self.original_labels.keys():
            labels_for_users["user_labels"][user_name] = {
                "labels": [],
                "true_labels": []
            }
        # now all the users' labels are loaded
        # should already handle train-test split here
        # words not in train set should be labeled as 0
        print("processing labels...")
       
 
        for sample in tqdm(samples, desc="creating labels"):
            tokens = sample["tokens"]
            words_in_sentence = sample["words_in_sentence"]
            
            # add user's labels for the current tokens
            temp_user_labels = {user_name: {
                "labels": [],
                "true_labels": []
            } for user_name in self.original_labels.keys()}
            for token in tokens:
                if token in words_in_sentence: # the token appears in the word_list
                    if token in self.train_words: # train set   
                        for user_name, labels in self.original_labels.items():
                            if labels[token] == 'NA':
                                train_label_to_add = 0 # no gradient for 0
                            else:
                                train_label_to_add = labels[token]
                            temp_user_labels[user_name]["labels"].append(train_label_to_add)
                            temp_user_labels[user_name]["true_labels"].append(labels[token])
                        
                    else: # test set
                        for user_name, labels in self.original_labels.items():
                            temp_user_labels[user_name]["labels"].append(-1) # test set is labeled as -1
                            temp_user_labels[user_name]["true_labels"].append(labels[token])

                else: # the token was not labeled by the user (not in the word_list)
                    for user_name, labels in self.original_labels.items():
                        temp_user_labels[user_name]["labels"].append(0) # non shown words are labeled as 0
                        temp_user_labels[user_name]["true_labels"].append(0)
            # clean samples that only contains 0 
            
    
            for user_name, labels in temp_user_labels.items():
                labels_for_users["user_labels"][user_name]["labels"].append(labels["labels"])
                labels_for_users["user_labels"][user_name]["true_labels"].append(labels["true_labels"])
            labels_for_users["tokens"].append(tokens)

        # ckeck that number of tokens is the same as the number of labels
        number_of_tokens = len(labels_for_users["tokens"])
        number_of_labels = len(labels_for_users["user_labels"]["user0"]["labels"])
        assert number_of_tokens == number_of_labels, "number of tokens is not the same as the number of labels"
        return labels_for_users

    def save_labels(self, labels, save_labels_path):
        print("saving labels...")
        with open(save_labels_path, "w", encoding="utf-8") as f:
            # Save each sample as a separate JSON line
            for i in tqdm(range(len(labels["tokens"])), desc="saving labels"):
                sample = {
                    "tokens": labels["tokens"][i],
                    "user_labels": {}
                }
                for user_name in labels["user_labels"]:
                    sample["user_labels"][user_name] = {
                        "labels": labels["user_labels"][user_name]["labels"][i],
                        "true_labels": labels["user_labels"][user_name]["true_labels"][i]
                    }
                f.write(json.dumps(sample) + "\n")

class SentenceLevelTokenLabeler:
    # each sentence contains a single unique word with its labels, 
    # so train-test split can be done at the sentence level
    # label structure: 
    #  {
    #      "word": word, 
    #      "sentence": list of sentences,
    #      "tokens": list of tokens,
    #      "labels": user_name: list of labels, only the word has label, other tokens are labeled as 0
    #  }
    def __init__(self, sentence_words, sentences):
        random.seed(0)  # Keep same seed as word split for consistency
        self.all_samples = []
        self.original_labels = {}
        self.word_list = list(set([word for words in sentence_words.values() for word in words]))
        self.final_labels = {
            word:{
                # "sentence": [],
                "tokens": [],
                "labels": []
            } for word in self.word_list
        }
        sentence_tokenizer = SentenceTokenizerFast()
        # Process each sentence
        max_length = 0
        min_length = 100
        for sentence_idx, words_in_sentence in tqdm(sentence_words.items(), desc="processing sentences"):
            # Get the sentence text and tokens
            sentence = sentences[int(sentence_idx)]
            tokens = sentence_tokenizer.tokenize_sentence(sentence, use_rule=True)
            tokens_length = len(tokens)
            if tokens_length > max_length:
                max_length = tokens_length
            if tokens_length < min_length:
                min_length = tokens_length
            if tokens_length<10 or tokens_length>50: # only keep sentences with 10-50 tokens; this should already be handled previously
                continue
            else:
                # Store tokens and words for this sentence
                self.all_samples.append({
                    "tokens": tokens,
                    "words_in_sentence": words_in_sentence,
                })
        print(f"max token length: {max_length}, min token length: {min_length}")
        # Randomly shuffle all samples
        random.shuffle(self.all_samples)

    def create_labels(self, original_labels_dir, max_samples_per_word=20): 
        """
        max_samples_per_word: how many sentences a word has with labels
        """
        for file in os.listdir(original_labels_dir):

            if file.endswith(".txt"):
                # beginning processing for each user 
                file_prefix = file.split(".")[0]
                if len(file_prefix.split("_")) > 1:
                    user_name = file_prefix.split("_")[-1]
                else:
                    user_name = file_prefix
                self.original_labels[user_name] = {}
                with open(os.path.join(original_labels_dir, file), "r", encoding="utf-8") as f:
                    # read all the labels for the current user
                    labels = [l for l in f.readlines() if l.strip("\n") != ""]
                    for line in labels:
                        word = line.strip("\n").split("\t")[0].lower()
                        label = line.strip("\n").split("\t")[1]
                        self.original_labels[user_name][word] = label

        for word_to_label in tqdm(self.word_list, desc="creating labels"):
            sample_count = 0
            for sample in self.all_samples:
                if word_to_label in sample["words_in_sentence"]: # word is in the sentence
                    # self.final_labels[word_to_label]["sentence"].append(sample["sentence"])
                    self.final_labels[word_to_label]["tokens"].append(sample["tokens"])
                    labeling_data = {}
                    for user_name, word_labels in self.original_labels.items():
                        if word_labels[word_to_label] == 'NA':
                            continue
                        else:
                            labeling_data[user_name] = []
        
                    for token in sample["tokens"]:
                        for user_name in labeling_data.keys():
                            user_word_labels = self.original_labels[user_name]
                        
                            if token == word_to_label:
                                labeling_data[user_name].append(user_word_labels[token])
                            else:
                                labeling_data[user_name].append(0)
                    self.final_labels[word_to_label]["labels"].append(labeling_data)
                    sample_count += 1
                if sample_count >= max_samples_per_word:
                    break
    def save_labels(self, save_labels_path):
        print("saving labels...")
        with open(save_labels_path, "w", encoding="utf-8") as f:
            for word in tqdm(self.word_list, desc="saving labels"):
                sample = self.final_labels[word]
                f.write(json.dumps({word: sample}) + "\n")
                           

if __name__ == "__main__":
    # TODO: wrap create labels and sentence processing, to a single entry: input texts and labels, output token labels
    OUTPUT_DIR = "path/to/generated_outputs/coca_simulation_llama/" # the dir to store the generated labels 
    FILE_DIR = "path/to/generated_outputs/coca/" # the corpus dir 
    DATA_DIR = "path/to/data/" # labels dir 
    sentence_words_file = os.path.join(FILE_DIR, "sentence_matching_words.json") # corpus file, in which stores the sentences that contain the target words
    sentences_file = os.path.join(FILE_DIR, "unique_idx_sentences.txt")

    # original_labels_dir = os.path.join(DATA_DIR, "random_vocab_knowledge")
    original_labels_dir  = "path/to/generated_inference_outputs/simulation_experiments/outputs/exp2/test_results_llama/organized"
    save_train_labels_path = os.path.join(OUTPUT_DIR, "train_labels.json")
    save_test_labels_path = os.path.join(OUTPUT_DIR, "test_labels.json")

    sentence_words = read_sentences_words(sentence_words_file)
    with open(sentences_file, "r", encoding="utf-8") as f:
        sentences = [line.strip("\n") for line in f.readlines()]
    
    # Process each sentence
    print(len(sentence_words))

    # token level labels: each sentence containing different words, some with masks
    # # token_labeler = TokenLabelerTrainTestSplit(sentence_words, sentences, use_rule_for_tokenization=False)
    # # token_labeler.create_labels(original_labels_dir, max_samples_per_word=100, test_k = 10)
    # # token_labeler.save_labels(token_labeler.train_labels, save_train_labels_path)
    # # token_labeler.save_labels(token_labeler.test_labels, save_test_labels_path)

    # create sentence level labels, each sentence only contains one single "labeled word"
    output_label_path = os.path.join(OUTPUT_DIR,"sentence_level_sim_labels.json")

    token_labeler = SentenceLevelTokenLabeler(sentence_words, sentences)
    token_labeler.create_labels(original_labels_dir, max_samples_per_word=5)
    token_labeler.save_labels(output_label_path)


    with open(output_label_path, "r") as f:
        word_data = []
        for line in tqdm(f.readlines()):
            word_data.append(json.loads(line))
    data = {}
    for d in word_data:
        for k, v in d.items():
            data[k] = v

    user_data = {}
    for word, label_data in data.items():
        for tokens, user_labels in zip(label_data['tokens'], label_data['labels']):
            for user_name, user_label in user_labels.items():
                if user_name not in user_data:
                    user_data[user_name] = {}
                user_data[user_name][word] = {
                    "tokens": [],
                    "labels": []
                }
                user_data[user_name][word]["tokens"].append(tokens)
                user_data[user_name][word]["labels"].append(user_label)
    result_user_data = {}
    for user_name, word_data in user_data.items():
        result_user_data[user_name] = []
        for word, data in word_data.items():
            result_user_data[user_name].append({
                "word": word,
                "tokens": data["tokens"],
                "labels": data["labels"]
            })
    output_label_path_per_user = os.path.join(OUTPUT_DIR,"sentence_level_sim_labels_per_user.json")
    with open(output_label_path_per_user, "w", encoding="utf-8") as f:
        for user_name, data in tqdm(result_user_data.items(), desc="saving labels"):
            f.write(json.dumps({user_name: data}) + "\n")
            
    # # with open(os.path.join(OUTPUT_DIR, "token_label_count.json"), "w", encoding="utf-8") as f:
    # #     json.dump(token_labeler.token_label_count, f)
    # token_labeler.save_labels(save_labels_path)



    # for tok, lab, true_lab   in zip(labels['tokens'][:2], labels['user_labels']['user9']['labels'][:2], labels['user_labels']['user9']['true_labels'][:2]):
    #     for t, l, tl in zip(tok, lab, true_lab):  
    #         print(t, l, tl)
    #     print("-"*100)

   


    # with open(os.path.join(OUTPUT_DIR, "token_label_count.json"), "r", encoding="utf-8") as f:
    #     token_label_count = json.load(f)
    #     # Sort token_label_count by values in descending order
    #     sorted_token_label_count = dict(sorted(token_label_count.items(), key=lambda x: x[1], reverse=True))
    #     # Get the value at 10th percentile
    #     values = list(sorted_token_label_count.values())
    #     percentile_10 = values[int(len(values) * 0.1)]
    #     print(f"10th percentile value: {percentile_10}") # 10th percentile value: 2000
        