# one episode contains a set of tasks 
# each task provides one support batch and one query batch from a learner 

# Task: Train and test split based on individuals 
# Each individual's data is split into support set and query set
# support: first randomly sample 10 initial words based on the word frequency bucket; then after one epoch, fetch next 10 words based on uncertainty/confidence error 
# In testing, we will first use 10 same initial words, then update the model with 10 samples each batch 

import random
import math
from collections import defaultdict
from wordfreq  import word_frequency  

class IndividualTask:
    """
    Handles the data for one individual (i.e. one task).

    Attributes:
        user_data (list): A list of word data dictionaries for one user.
        support_set (list): The support set obtained after splitting the user_data.
        query_set (list): The query set obtained after splitting the user_data.
    """
    def __init__(self, user_data, support_ratio=0.3):
        self.user_data = user_data  # All word samples for the user.
        self.support_set, self.query_set = self.split_data(support_ratio)
        self.data_size = len(self.user_data)

    def split_data(self, support_ratio):
        """
        Split the user's data into support and query sets.
        The split is based on words, using the provided support_ratio (e.g., 0.3 means 30% support and 70% query).
        """
        # Shuffle the data to randomize the split.
        data = self.user_data.copy()
        random.Random(0).shuffle(data) # support/query split is fixed 
        if support_ratio <= 1:
            split_idx = int(len(data) * support_ratio)
            support = data[:split_idx]
            query = data[split_idx:]
        else:
            if type(support_ratio) is not int:
                raise ValueError("support_ratio larger than 1 must be an integer")
            support = data[:support_ratio]
            query = data[support_ratio:]
        return support, query

    def get_frequency_band(self, word, num_bands=10):
        """
        Compute a frequency band for the given word.
        This function uses the `word_frequency` library to get the frequency and then
        assigns the word to a bucket (band). Here we use a logarithmic scale for bucketing.
        """
        freq = word_frequency(word, 'en')  # returns frequency as a float
        # The bucket is computed by taking the log, capped to num_bands-1.
        if freq == 0:
        # Handle out-of-vocab or extremely rare words
            band = num_bands - 1
        else:
            # Use -log10 to convert low-frequency words to higher bucket numbers
            log_freq = -math.log10(freq)
            # Normalize to a fixed number of bands (e.g., 0-9)
            band = min(int(log_freq), num_bands - 1)
        return band

    def sample_support(self, num_samples=10):
        """
        Sample support examples from the support_set.
        Initially, samples are selected by bucketing words by their frequency,
        ensuring each frequency band is represented.
        """
        buckets = defaultdict(list)
        # Place each sample in a frequency bucket.
        for word_data in self.support_set:
            band = self.get_frequency_band(word_data["word"])
            buckets[band].append(word_data)
        
        support_samples = []
        bucket_keys = list(buckets.keys())
        num_buckets = len(bucket_keys)
        # If there are at least as many buckets as samples, sample one from each selected bucket.
        if num_buckets >= num_samples:
            chosen_buckets = random.sample(bucket_keys, num_samples)
            for b in chosen_buckets:
                support_samples.append(random.choice(buckets[b]))
        else:
            # First, pick one sample from each bucket.
            for b in bucket_keys:
                support_samples.append(random.choice(buckets[b]))
           

            remaining = num_samples - len(support_samples)
       
            if remaining == 0:
                return support_samples
            # Then, fill the rest by randomly sampling across all buckets.
            all_samples = [sample for bucket in buckets.values() for sample in bucket]
           
            support_samples.extend(random.sample(all_samples, remaining))
        
        return support_samples

    def sample_query(self, random_sample = True, num_samples=100):
        """
        Randomly sample query examples from the query_set;
        or sample based on the frequency of the words in the query set
        """
        if random_sample:
            if len(self.query_set) >= num_samples:
                return random.sample(self.query_set, num_samples)
            else:
                # If there are not enough examples, return all or consider sampling with replacement.
                return self.query_set
        else:
            # Bucket the query_set by frequency band
            buckets = defaultdict(list)
            for word_data in self.query_set:
                band = self.get_frequency_band(word_data["word"])
                buckets[band].append(word_data)

            query_samples = []
            bucket_keys = list(buckets.keys())
            num_buckets = len(bucket_keys)

            if num_buckets >= num_samples:
                # Enough bands to sample one from each
                chosen_buckets = random.sample(bucket_keys, num_samples)
                for b in chosen_buckets:
                    query_samples.append(random.choice(buckets[b]))
            else:
                # Sample one from each band first
                for b in bucket_keys:
                    query_samples.append(random.choice(buckets[b]))

                remaining = num_samples - len(query_samples)
                if remaining == 0:
                    return query_samples

                # Fill the rest from all available query_set examples in buckets
                all_samples = [sample for bucket in buckets.values() for sample in bucket]
                if remaining <= len(all_samples):
                    query_samples.extend(random.sample(all_samples, remaining))
                else:
                    query_samples.extend(all_samples)  # not enough unique left

            return query_samples
      

    def select_next_samples(self, model, num_samples):
        raise NotImplementedError()
    
    def get_sample_by_word(self, word):
        for sample in self.user_data:
            if sample["word"].lower().strip() == word.lower().strip():
                return sample
        return None


class EpisodeLoader:
    """
    Constructs an episode consisting of multiple tasks.

    Each episode is a set of tasks, and each task provides a support batch and a query batch.
    """
    def __init__(self, tasks_data, support_ratio=0.3, num_support=10, num_query=100):
        """
        Args:
            tasks_data (dict): Mapping from user name to list of word data dictionaries.
            support_ratio (float): Ratio of data to use as support.
            num_support (int): Number of support samples per task.
            num_query (int): Number of query samples per task.
        """
        self.tasks_data = tasks_data
        self.support_ratio = support_ratio
        self.num_support = num_support
        self.num_query = num_query

    def get_episode_tasks(self, task_names=None):
        """
        Constructs an episode by fetching tasks.
        
        Args:
            task_names (list, optional): List of user names to include in the episode.
                If None, all tasks in tasks_data are used.
                
        Returns:
            list: A list where each element is a tuple (support_batch, query_batch) for a task.
        """
        if task_names is None:
            task_names = list(self.tasks_data.keys())
        
        episode_tasks = []
        for user in task_names:
            task = IndividualTask(self.tasks_data[user], self.support_ratio)
            support_batch = task.sample_support(self.num_support)
            query_batch = task.sample_query(self.num_query)
            episode_tasks.append((support_batch, query_batch))
        
        return episode_tasks

if __name__ == "__main__":
    import json
    from tqdm import tqdm
    # with open("path/to/generated_outputs/coca_simulation/sentence_level_labels.json", "r") as f:
    #     word_data = []
    #     for line in tqdm(f.readlines()):
    #         word_data.append(json.loads(line))
    # data = {}
    # for d in word_data:
    #     for k, v in d.items():
    #         data[k] = v
    # # print(data['list']['labels'][0])
    # # print(data['list']['tokens'][0])
    # # print(len(data['list']['labels']))
    # # print(len(data['list']['tokens']))
    # user_data = {}
    # for word, label_data in data.items():
    #     for tokens, user_labels in zip(label_data['tokens'], label_data['labels']):
    #         for user_name, user_label in user_labels.items():
    #             if user_name not in user_data:
    #                 user_data[user_name] = {}
    #             user_data[user_name][word] = {
    #                 "tokens": [],
    #                 "labels": []
    #             }
    #             user_data[user_name][word]["tokens"].append(tokens)
    #             user_data[user_name][word]["labels"].append(user_label)
    # result_user_data = {}
    # for user_name, word_data in user_data.items():
    #     result_user_data[user_name] = []
    #     for word, data in word_data.items():
    #         result_user_data[user_name].append({
    #             "word": word,
    #             "tokens": data["tokens"],
    #             "labels": data["labels"]
    #         })
    # for user_name, word_data in result_user_data.items():
    #     task = IndividualTask(word_data)
    #     print(task.support_set[0])
    #     print(task.query_set[0])
    #     print(len(task.support_set))
    #     print(len(task.query_set))
    #     break
    
    user_data = []
    with open("path/to/generated_outputs/coca_simulation/sentence_level_user_labels.json", "r") as f:
        for line in tqdm(f.readlines()):
            user_data.append(json.loads(line))
    task = IndividualTask(list(user_data[0].values()))
    support_batch = task.sample_support(10)
    for s in support_batch:
        print(s)
        