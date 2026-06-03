# train with word_level split, one sentence contain different word labels
# deprecated; use sent level


import numpy as np
from datasets import Dataset
# import evaluate
# from seqeval.metrics import accuracy_score, classification_report
import os
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, BertForTokenClassification
import torch
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer

import json
from sklearn.metrics import classification_report
from tqdm import tqdm
import wandb
from functools import partial
import argparse

from data_utils.read_resources import load_labels
label_list = {
    -1: -100,
    0 : -100,
    1 : 0,
    2:  0, 
    3:  0,
    4:  1, 
    5:  1

}
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a BERT model for token classification")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Pretrained transformer model name or path")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="Number of token labels")
    
    # Data arguments
    parser.add_argument("--train_labels_path", type=str, required=True,
                        help="Path to train set token labels")
    parser.add_argument("--test_labels_path", type=str, required=True,
                        help="Path to test set token labels")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=256,
                        help="Evaluation batch size")
    

    # Training arguments
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for optimizer")
    
    # Output and logging
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save model checkpoints")
    
    parser.add_argument("--cuda", type=str, default="all",
                        help="cuda device: 0, 1, 2, 3...; default is all")
    
    return parser.parse_args()

def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    
    labels = []
    original_labels = []
    labels_with_test_tokens = []
    for i, (label, original_true_label) in enumerate(zip(examples['labels'], examples['true_labels'])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        original_ids = []
        true_ids  = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
                original_ids.append(-100)
                true_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                mapped_label = label_list[label[word_idx]]
                label_ids.append(mapped_label) # what model will learn
                original_ids.append(label[word_idx]) # include the position of test tokens 
                true_ids.append(original_true_label[word_idx]) # include the real label of test tokens
            else:
                label_ids.append(-100)
                original_ids.append(-100)
                true_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        original_labels.append(original_ids)
        labels_with_test_tokens.append(true_ids)
    tokenized_inputs["labels"] = labels
    tokenized_inputs["original_labels"] = original_labels
    tokenized_inputs["true_labels"] = labels_with_test_tokens
    return tokenized_inputs

class CustomTrainer(Trainer):
    # custom the trainer evaluate 
    def evaluate(self, eval_dataset=None, ignore_keys=None):
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        predictions = self.predict(eval_dataset).predictions
        predictions = np.argmax(predictions, axis=-1)

        true_entities = []
        pred_entities = []
        # orig_label: include -1 to indicate test token position; 
        # label - true_labels: include the user labeling for test tokens;
        for pred, label, orig_label in tqdm(zip(predictions, eval_dataset["true_labels"], eval_dataset["original_labels"]), desc="Evaluating"):
            true_seq = []
            pred_seq = []

            for p_token, l_token, orig_token in zip(pred, label, orig_label):
                if orig_token == -1:  # Only evaluate test labels
                    true_seq.append(label_list[l_token])
                    pred_seq.append(p_token)

            true_entities.append(true_seq)
            pred_entities.append(pred_seq)
        y_true= np.array([item for sublist in true_entities for item in sublist])
        y_pred= np.array([item for sublist in pred_entities for item in sublist])

        results = classification_report(y_true, y_pred, labels=[0,1], digits=4, output_dict=True) # set output_dict=True when logging
        wandb.log(results)
        return results

def main():
    args = parse_args()
    if args.cuda == "all":
        pass
    else:
        print(args.cuda)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        print(f"currently using { torch.cuda.device_count()} GPUs")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print("loading labels...")
    train_set_labels = load_labels(args.train_labels_path)
    test_set_labels = load_labels(args.test_labels_path)
    print("finished loading labels")
    train_set_token_list = train_set_labels['tokens']
    test_set_token_list = test_set_labels['tokens']
    tokenize_function = partial(tokenize_and_align_labels, tokenizer)
    # ds_list = {}
    # run_cnt = 0
    for users in train_set_labels['user_labels'].keys():
        # if run_cnt == 0:
        #     run_cnt += 1
        #     continue
        print("training for current user: ", users)
        train_labels = train_set_labels['user_labels'][users]['labels']
        test_labels = test_set_labels['user_labels'][users]['labels']
        train_true_labels = train_set_labels['user_labels'][users]['true_labels']
        test_true_labels = test_set_labels['user_labels'][users]['true_labels']
        train_labels = [[int(l) for l in label] for label in train_labels]
        train_true_labels = [[int(l) for l in true_label] for true_label in train_true_labels]
        test_labels = [[int(l) for l in label] for label in test_labels]
        test_true_labels = [[int(l) for l in true_label] for true_label in test_true_labels]
        
        train_ds = Dataset.from_dict({
            "tokens":train_set_token_list,
            "labels":train_labels,
            "true_labels":train_true_labels,
        })
        test_ds = Dataset.from_dict({       
            "tokens":test_set_token_list,
            "labels":test_labels,
            "true_labels":test_true_labels,
        })
        current_train_ds = train_ds
        current_test_ds = test_ds

        wandb.init(project="vocab-predict-full_training-bert-2", name=users)
        train_dataset_tokenized = current_train_ds.map(tokenize_function, batched=True)
        test_dataset_tokenized = current_test_ds.map(tokenize_function, batched=True)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            save_strategy="no",
            report_to = 'wandb',
            fp16=True,
            # load_best_model_at_end=True,
        )
        model = BertForTokenClassification.from_pretrained(args.model_name, num_labels=args.num_labels) # initialize a new model for each user
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_tokenized,
            eval_dataset=test_dataset_tokenized,
            data_collator=DataCollatorForTokenClassification(tokenizer),
        )
        trainer.train()
        trainer.evaluate()
        wandb.finish()
if __name__ == "__main__":
    main()
 
