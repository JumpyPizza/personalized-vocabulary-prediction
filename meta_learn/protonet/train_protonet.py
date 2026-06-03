import sys
sys.path.append("../")
sys.path.append("../../llm_inference")
sys.path.append("../meta_learn")
sys.path.append("../../AL")

import os 
from typing import List, Dict, Any, Optional, Tuple
import json
import random
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report
from tqdm import tqdm 
import wandb 

from transformers import AutoTokenizer
import torch

from protonet import ProtoNetWrapper, UserProtoConfig
from bert_model import BertTokenTask   


def evaluate_episode_userproto(
    tasks: List[BertTokenTask],
    model: ProtoNetWrapper,
    mix_neighbors: bool = True,
    print_report: bool = True
) -> Tuple[float, float, float]:
    f1s, accs, baccs = [], [], []
    for task in tqdm(tasks, desc="evaluating"):
        preds, labels = model.evaluate_task(task, mix_neighbors=mix_neighbors)
        if len(labels) == 0:
            continue
        y_true = np.array(labels)
        y_pred = np.array(preds)
        if print_report:
            print(classification_report(y_true=y_true, y_pred=y_pred, digits=4))
        f1s.append(f1_score(y_true, y_pred, average="macro"))
        accs.append(accuracy_score(y_true, y_pred))
        baccs.append(balanced_accuracy_score(y_true, y_pred))
    return float(np.mean(f1s)), float(np.mean(accs)), float(np.mean(baccs))


def build_tasks(user_dataset: List[Dict[str, Any]], tokenizer: AutoTokenizer, support_ratio: float, task_prefix: str) -> List[BertTokenTask]:
    tasks = []
    for data in user_dataset:
        t = BertTokenTask(data, tokenizer, support_ratio=support_ratio)
        # ensure each task has an id for memory; use provided field if exists
        if not hasattr(t, "task_id"):
            # try to pull a user name if present in your user_data; otherwise hash object
            setattr(t, "task_id", task_prefix+"_"+str(id(t)))
        tasks.append(t)
    return tasks


def train_userproto(
    model_name: str,
    tokenizer: AutoTokenizer,
    train_user_data: List[Dict[str, Any]],
    test_user_data: List[Dict[str, Any]],

    iteration_num: int,
    task_num_in_episode: int,
    train_support_batch_size: int = 30,
    train_query_batch_size: int = 100,
    test_support_size = 30, 
    cfg: Optional[UserProtoConfig] = None,
    wandb_run=None,
    save_model_path: Optional[str] = None,
    save_steps = 50,
    eval_steps = 50,
    print_report_for_test = False,
    
    
):
    """
    Training loop similar to your Reptile setup, but using ProtoUserWrapper.
    """
    cfg = cfg or UserProtoConfig()
    model = ProtoNetWrapper(model_name=model_name, cfg=cfg, wandb_run=wandb_run)

   
    train_users = train_user_data
    test_users = test_user_data


    best_f1 = 0.0

    # the train_support_ratio here defines how many data will be exposed and be available for sampling in the support pool
    train_task_pool = build_tasks(train_users, tokenizer, support_ratio=0.5, task_prefix = "train") 
    test_task_pool = build_tasks(test_users, tokenizer, support_ratio=test_support_size, task_prefix = "test") 
    for it in range(iteration_num):
        # build an episode of user tasks
        train_tasks = random.sample(train_task_pool, min(len(train_task_pool), task_num_in_episode) )

        # meta update
        # the support batch size and query batch size is used during meta-learning
        logs = model.meta_update(train_tasks, support_batch_size=train_support_batch_size, query_batch_size=train_query_batch_size)
        if wandb_run is not None:
            wandb_run.log({**logs, "iter": it})
        else:
            print(f"-------iter: {it}-------, loss {logs['meta/ce']}")

        # periodic eval
        if it % eval_steps == 0:
            print(f"DEBUG EVAL TRIGGER at it={it}")
            eval_tasks = test_task_pool
            f1, acc, bacc = evaluate_episode_userproto(
                eval_tasks, model, mix_neighbors=True,
                print_report = print_report_for_test
            )
            if wandb_run is not None:
                wandb_run.log({"test/f1": f1, "test/acc": acc, "test/bacc": bacc, "iter": it})
                print(f"[it={it}] f1={f1:.4f} acc={acc:.4f} bacc={bacc:.4f}")
            else:
                print(f"[it={it}] f1={f1:.4f} acc={acc:.4f} bacc={bacc:.4f}")
            if save_model_path is not None and save_steps == "best":
                if f1 > best_f1:
                    best_f1 = f1
                    model.save(os.path.join(save_model_path, f"best_{it}.ckpt"))
        if save_model_path is not None:
            if save_steps != "best":
                if (it+1) % save_steps == 0:
                    print(f"saving at iter {it}")
                    model.save(os.path.join(save_model_path, f"{it}.ckpt"))

    return model

if __name__ == "__main__":
    model_path = "path/to/model_checkpoints/roberta-base/"
    data_path = "path/to/data/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)

    user_file_path = data_path + "sentence_level_real_user_labels.json"

    human_user_data = []
    with open(user_file_path, "r") as f:
        
        for idx, line in enumerate(tqdm(f.readlines())):
            human_user_data.append(json.loads(line))
            # if idx > 2:
            #     break
    print(f"human user data size: {len(human_user_data)}")

    simulation_data = []
    simulation_user_data = []

    user_sim_json_fn = data_path + "qwen_sentence_level_user_labels.json"
    N = 500
    reservoir = []
    with open(user_sim_json_fn, 'r') as f:
        for i, line in enumerate(tqdm(f)):
          
            if i < N:
                reservoir.append(line)
            else:
                r = random.randint(0, i)
                if r < N:
                    reservoir[r] = line

    # Now load the sampled lines into JSON
    for idx, line in enumerate(tqdm(reservoir)):
        line_data = json.loads(line)
        if any("user" in key for key in line_data.keys()):
            print("skipping ", line_data.keys())
            continue
        simulation_data.append(line_data)
    user_names = [list(u.keys())[0] for u in simulation_data]

    for name, data in zip(user_names, simulation_data):
        simulation_user_data.append(data[name]) # one element is a list of data
    print(f"simulation user data length : {len(simulation_user_data)}")
    train_user_data = simulation_user_data



    test_user_data = human_user_data

    iteration_num = 1000
    task_num_in_episode = 30
    
    cfg = UserProtoConfig(encoder_lr = 2e-5, embedder_lr = 1e-4, lam=0.5, k_neighbors = 10)
    # save_path = "path/to/trained_checkpoints/proto_bert_base_2ndrun/"
    save_path = None
    wandb_run = wandb.init(project="protonet_vocab_predict", name="roberta-base-test-12k-lam-0.5")
    train_userproto(model_path, tokenizer, train_user_data, test_user_data, 
                    iteration_num = iteration_num, task_num_in_episode = task_num_in_episode,
                    wandb_run = wandb_run, cfg = cfg, save_model_path = save_path,
                    train_support_batch_size = 250, test_support_size = 50,
                    save_steps = 100, eval_steps = 50, print_report_for_test = True)


