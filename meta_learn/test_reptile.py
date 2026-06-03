# load the saved ckpt during training 
# do inference only 

import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from reptile import ReptileWrapper
from bert_model import BertTokenClf, BertTokenTask


def evaluate_user(reptile, base_model_path, user_data, tokenizer, support_size, inner_steps, num_runs=5, batch_size=100):
    """Run multiple adaptation-evaluation runs for a single user task."""
    accs, f1s, mccs = [], [], []

    for run in range(num_runs):
        # Create a fresh task for this run
 
        task = BertTokenTask(user_data, tokenizer, support_ratio=support_size) 

        # Reset model to ckpt each run
        reptile.load_model(base_model_path)

        # ===== Adaptation phase =====
        support_input, _, _ = task.get_data(support_batch_size=support_size, query_batch_size=0) # support_size usually less than 50 
        for _ in range(inner_steps):
            reptile.meta_step([support_input], max_support_subbatch_size=100)

        # ===== Evaluation phase =====
        eval_inputs = task.get_evaluation_set(task.query_set)
        preds, labels = reptile.evaluate(eval_inputs, batch_size)

        accs.append(accuracy_score(labels, preds))
        f1s.append(f1_score(labels, preds, average="macro"))
        mccs.append(matthews_corrcoef(labels, preds))

    return {
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "f1_mean": np.mean(f1s),   "f1_std": np.std(f1s),
        "mcc_mean": np.mean(mccs), "mcc_std": np.std(mccs)
    }


def load_test_data(json_path):
    user_data = {}
    with open(json_path, "r") as f:
        for line in tqdm(f, desc="loading user file"):
            d = json.loads(line)
            for user, entries in d.items():
                user_data[user] = entries
    return user_data


def main(args):
    # Load model & tokenizer
    model = BertTokenClf(args.model_path, args.num_labels).to("cuda")
    if "roberta" in args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, max_length=512, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, max_length=512)

    reptile = ReptileWrapper(model, args.inner_lr, args.outer_lr, args.inner_steps, wandb_run=None)
    # reptile.load_model(args.load_model_path)

    # Prepare test data
    test_users = load_test_data(args.test_data_path)

    results = []
    for user, entries in tqdm(test_users.items(), desc="Testing users"):
        metrics = evaluate_user(reptile, args.load_model_path, entries, tokenizer,
                                support_size=args.adaptation_size,
                                inner_steps=args.inner_steps,
                                num_runs=args.num_runs,
                                batch_size=args.eval_batch_size)
        results.append({"user": user, **metrics})
        # break
    df = pd.DataFrame(results)
    # print(df)

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_path, index=False)
        print(f"Results saved to {args.output_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--num_labels", type=int, default=2)
#     parser.add_argument("--load_model_path", type=str, required=True)
#     parser.add_argument("--test_data_path", type=str, required=True)
#     parser.add_argument("--output_path", type=str, default=None)

#     parser.add_argument("--adaptation_size", type=int, default=20) # the size of the support data
#     parser.add_argument("--num_runs", type=int, default=5)
#     parser.add_argument("--eval_batch_size", type=int, default=100)


#     parser.add_argument("--inner_lr", type=float, default=1e-5)
#     parser.add_argument("--outer_lr", type=float, default=0.0)
#     parser.add_argument("--inner_steps", type=int, default=5)

#     args = parser.parse_args()

#     main(args)

if __name__ == "__main__":
    class Args:
        model_path = "path/to/model_checkpoints/deberta-v3-large/"
        load_model_path = "path/to/trained_checkpoints/deberta-v3-large/best_ckpt.pt"
        
        num_labels = 2
        
        test_data_path = "path/to/data/sentence_level_evkd_user_labels.jsonl"
        output_path = "./test_metrics.csv"
        # output_path = None
        adaptation_size = 50
        num_runs = 5
        eval_batch_size = 100
        inner_lr = 1e-5 # smaller model: 1e-5; larger model 1e-6/5e-6
        outer_lr = 0.0 #placeholder, not used
        inner_steps = 2 # in testing
    # args = Args.model_path
    # model_list = ["bert-base-uncased",  "roberta-base",  "deberta-v3-base" ] 
    # ckpt_list = ["bert-base", "roberta-base",  "deberta-v3-base" ] 
    # model_list = ["bert-large-uncased","roberta-large"] # different lr 
    # ckpt_list = ["bert-large", "roberta-large"]
    model_list = ["roberta-large"] # different lr 
    ckpt_list = ["roberta-large"]
  
    for model, ckpt in zip(model_list, ckpt_list):
        print(ckpt)
        model_path = f"path/to/model_checkpoints/{model}/"
        load_model_path = f"path/to/trained_checkpoints/{ckpt}/best_ckpt.pt"
        Args.model_path = model_path
        Args.load_model_path = load_model_path
        Args.output_path = f"./{ckpt}_metrics.csv"
        main(Args())
        
    # main(Args())