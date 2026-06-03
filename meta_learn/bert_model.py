from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from task_loader import IndividualTask
import random
from sklearn.metrics import f1_score
import numpy as np

label_list = {
    -1: -100,
    0 : -100,
    1 : 0,
    2:  0, 
    3:  0,
    4:  1, 
    5:  1

}
def tokenize_and_align_labels(tokenizer, token_lists, labels_of_tokens):
    # Tokenize with padding and truncation
    tokenized_inputs = tokenizer(
        token_lists,
        is_split_into_words=True,
        return_tensors="pt",          # Return PyTorch tensors
        padding=True,                 # Pad to the longest in the batch
        truncation=True
    )

    all_labels = []
    all_original_labels = []

    for i, label in enumerate(labels_of_tokens):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        original_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
                original_ids.append(-100)
            elif word_idx != previous_word_idx:
                mapped_label = label_list[int(label[word_idx])]
                label_ids.append(mapped_label)
                original_ids.append(int(label[word_idx]))
            else:
                label_ids.append(-100)
                original_ids.append(-100)

            previous_word_idx = word_idx

        all_labels.append(label_ids)
        all_original_labels.append(original_ids)

    # Pad label lists manually to match tokenized input length
    max_len = tokenized_inputs["input_ids"].shape[1]

    def _pad_to_max(seq, pad_value=-100):
        return seq + [pad_value] * (max_len - len(seq))

    all_labels = [_pad_to_max(seq) for seq in all_labels]
    all_original_labels = [_pad_to_max(seq) for seq in all_original_labels]

    tokenized_inputs["labels"] = torch.tensor(all_labels)
    tokenized_inputs["original_labels"] = torch.tensor(all_original_labels)

    return tokenized_inputs

class BertTokenTask(IndividualTask):
    def __init__(self, user_data, tokenizer, support_ratio=0.3, upsample= False):
        super().__init__(user_data, support_ratio)
        self.tokenizer = tokenizer
        self.upsample = upsample
        

    def sample_support_until_non_zero(self, support_batch_size=10):
        support_batch = self.sample_support(support_batch_size) # each call of get_data() will sample random data from fixed support/query set
        train_tokens = []
        train_labels = []
        for s in support_batch: # each sample in support_batch has N sentences, here we sample one each time
            random_index = random.randint(0, len(s['tokens']) - 1)
            sample_tokens = s['tokens'][random_index]
            sample_labels = s['labels'][random_index]
            train_tokens.append(sample_tokens)
            train_labels.append(sample_labels)
        sequence_labels = []
        for label_seq in train_labels:
            # Find first non-zero label in sequence
            non_zero = next((int(l) for l in label_seq if int(l) != 0 and int(l) != -1), 0)
            sequence_labels.append(non_zero)
     
        # Count samples in each category
        low_level = sum(1 for l in sequence_labels if l in [1,2,3]) 
        high_level = sum(1 for l in sequence_labels if l in [4,5])
        min_level = min(low_level, high_level)
        if min_level < support_batch_size*0.2:
            return False
        else:
            return support_batch

    def get_data(self, support_batch_size=10, query_batch_size=100, ensure_non_zero=False):
        # randomly sample support and query set based on frequency
        # if ensure_non_zero is True, then sample until set is balanced 
        # if self.upsample, then upsample the class with less samples to balance 
        query_batch = self.sample_query(query_batch_size)
        query_tokens = []
        query_labels = []

        if ensure_non_zero: # ensure that the support set has both labels by repeating sampling
            sample_cnt = 0
            while sample_cnt < 5000:
                support_batch = self.sample_support_until_non_zero(support_batch_size)
                sample_cnt += 1
                if not support_batch:
                    continue
                else:
                    # print("current sample cnt: ", sample_cnt)
                    break
            if not support_batch:
                support_batch = self.sample_support(support_batch_size)
        else:
            support_batch = self.sample_support(support_batch_size)
        train_tokens = []
        train_labels = []
        support_word_labels = []
        support_word_words = []
        for s in support_batch: # each sample in support_batch has N sentences, here we sample one each time
            random_index = random.randint(0, len(s['tokens']) - 1)
            
            sample_tokens = s['tokens'][random_index]
            sample_labels = s['labels'][random_index]
            sample_label_array = np.array(sample_labels) 
            if len(np.unique(sample_label_array[sample_label_array != 0])) == 0:
                cnt = 0 
                while cnt < 10:
                    random_index = random.randint(0, len(s['tokens']) - 1)
                    sample_tokens = s['tokens'][random_index]
                    sample_labels = s['labels'][random_index]
                    cnt += 1
                    sample_label_array = np.array(sample_labels) 
                    if len(np.unique(sample_label_array[sample_label_array != 0]))>0:
                        break
            if len(np.unique(sample_label_array[sample_label_array != 0])) == 0: 
                continue 
            train_tokens.append(sample_tokens)
            train_labels.append(sample_labels)
            ###### extract the label of this word ######
            sample_labels = [int(l) for l in sample_labels]
            sample_label_array = np.array(sample_labels) 
            label = np.unique(sample_label_array[sample_label_array != 0])[0]
            support_word_labels.append(int(label))
            support_word_words.append(s['word'])
        support_words = (support_word_words, support_word_labels)
        if self.upsample:
            train_tokens, train_labels = self.upsample_data(train_tokens, train_labels)

        for q in query_batch:
            random_index = random.randint(0, len(q['tokens']) - 1)
            sample_tokens = q['tokens'][random_index]
            sample_labels = q['labels'][random_index]
            query_tokens.append(sample_tokens)
            query_labels.append(sample_labels)
        support_input = tokenize_and_align_labels(self.tokenizer, train_tokens, train_labels)
        query_input = tokenize_and_align_labels(self.tokenizer, query_tokens, query_labels)
        return support_input, query_input, support_words # support_words is a tuple now, 0 is word, 1 is label
    
    def get_data_by_samples(self, samples):
        # given samples, return inputs that can be consumed by reptile
        train_tokens = []
        train_labels = []
        for s in samples: # each sample in support_batch has N sentences, here we sample one each time
            random_index = random.randint(0, len(s['tokens']) - 1)
            sample_tokens = s['tokens'][random_index]
            sample_labels = s['labels'][random_index]
            train_tokens.append(sample_tokens)
            train_labels.append(sample_labels)
        support_input = tokenize_and_align_labels(self.tokenizer, train_tokens, train_labels)
        return support_input
    
  
    
    def upsample_data(self, tokens, labels):
        # upsample the class with less samples to balance 
        # Get non-zero label for each sequence
        sequence_labels = []
        for label_seq in labels:
            # Find first non-zero label in sequence
            non_zero = next((int(l) for l in label_seq if int(l) != 0 and int(l) != -1), 0)
            sequence_labels.append(non_zero)
        print(sequence_labels)
        # Count samples in each category
        low_level = sum(1 for l in sequence_labels if l in [1,2,3]) 
        high_level = sum(1 for l in sequence_labels if l in [4,5])
        print("low_level: ", low_level, "high_level: ", high_level)
        # Determine which category needs upsampling
        if low_level == 0 or high_level == 0:
            return tokens, labels
            

        # Create lists of indices for each category
        low_indices = [i for i,l in enumerate(sequence_labels) if l in [1,2,3]]
        high_indices = [i for i,l in enumerate(sequence_labels) if l in [4,5]]
        
        # Upsample smaller category
        if low_level < high_level:
            indices_to_sample = low_indices
            num_samples = high_level - low_level
        else:
            indices_to_sample = high_indices  
            num_samples = low_level - high_level
            
        # Randomly sample indices with replacement
        sampled_indices = random.choices(indices_to_sample, k=num_samples)
        
        # Add upsampled data
        upsampled_tokens = tokens.copy()
        upsampled_labels = labels.copy()
        
        for idx in sampled_indices:
            upsampled_tokens.append(tokens[idx])
            upsampled_labels.append(labels[idx])
        
        # temp = []
        # for label_seq in upsampled_labels:
        #     # Find first non-zero label in sequence
        #     non_zero = next((l for l in label_seq if l != 0 and l != -1), 0)
        #     temp.append(non_zero)
        # print("the upsampled labels: ", temp)
        return upsampled_tokens, upsampled_labels
        
    def get_evaluation_set(self, task_set):
        evaluation_tokens = []
        evaluation_labels = []
        for data in task_set:
            random_index = random.randint(0, len(data['tokens']) - 1)
            evaluation_tokens.append(data['tokens'][random_index])
            evaluation_labels.append(data['labels'][random_index])
        evaluation_input = tokenize_and_align_labels(self.tokenizer, evaluation_tokens, evaluation_labels)
        return evaluation_input

    def select_next_samples(self, model, num_samples, metric, evaluate_batch_size=200):
        if metric == "entropy":
            # select samples based on the model uncertainty 
            full_query_set = self.get_evaluation_set(self.query_set) 
            logits = model.get_logits(full_query_set, evaluate_batch_size)
        
        
            # Calculate entropy as uncertainty measure
            probs = torch.softmax(torch.tensor(logits), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            # Get indices of samples with highest entropy
            top_indices = torch.topk(entropy, num_samples).indices
            selected_set = []
            for idx in top_indices:
                selected_set.append(self.query_set[idx])
        
            # update support set and query set
            new_support_set = self.support_set + selected_set
            self.support_set = new_support_set
            self.query_set = [sample for i, sample in enumerate(self.query_set) if i not in top_indices]
        elif metric == "frequency":
            # select samples based on the frequency of the tokens in the query set
            full_query_set = self.get_evaluation_set(self.query_set) 
            # TODO: implement frequency based sampling using sample_query(random=False)



class BertTokenClf(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        if not torch.cuda.is_available():
            raise ValueError("No GPU found")
        self.bert = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input):
        input_ids = input["input_ids"]
        attention_mask = input["attention_mask"]
        labels = input.get("labels", None)
        if labels is not None:
            labels = labels.to(self.bert.device)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs
    
    
        
    def evaluate_prediction(self, input, batch_size = 1000, return_logits=False, return_representation=False):
        tensor_dataset = TensorDataset(
            input["input_ids"],
            input["attention_mask"],
            input["labels"]
        )
        data_loader = DataLoader(tensor_dataset, batch_size=batch_size)
        pred_labels = []
        true_labels = []
        pred_logits = []
        representations = [] if return_representation else None
        final_indices = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.bert.device)
                attention_mask = attention_mask.to(self.bert.device)
                labels_np = labels.cpu().numpy()
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=return_representation)
                if return_representation:
                    last_hidden_states = outputs.hidden_states[-1] # [batch_size, seq_len, hidden_size]
                logits = outputs.logits
                pred = torch.argmax(logits, dim=2).cpu().numpy()
                logits_np = logits.cpu().numpy()
           
                for idx_in_batch, label_seq in enumerate(labels_np): # iterate over batch
                    pred_seq = pred[idx_in_batch]
                    logits_seq = logits_np[idx_in_batch]
                    assert len(pred_seq) == len(label_seq)
                
                    try:
                        label_index = np.where(label_seq != -100)[0][0] # from original label, get the position of the token 
                        word_label = label_seq[label_index] # original label
                        word_pred_label = pred_seq[label_index] # predicted label
                        pred_labels.append(word_pred_label)
                        true_labels.append(word_label)
                        pred_logits.append(logits_seq[label_index])
                        if return_representation:
                            vec = last_hidden_states[idx_in_batch,label_index]
                            representations.append(vec.cpu().numpy())
                        absolute_index = batch_idx * batch_size + idx_in_batch
                        final_indices.append(absolute_index)
                    # skips the missing data
                    except IndexError:
                        print("no labels in training data!")
                
        if not return_representation:
            if return_logits:
                return pred_labels, true_labels, pred_logits, final_indices
            else:
                return pred_labels, true_labels
        else:
            return pred_labels, true_labels, np.vstack(representations)
