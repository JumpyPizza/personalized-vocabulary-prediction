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

from transformers import AutoTokenizer
import torch

from protonet import ProtoNetWrapper, UserProtoConfig
from bert_model import BertTokenTask   
from train_protonet import evaluate_episode_userproto, build_tasks

data_path = "path/to/data/"
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
N = 300
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



base_model_path = "path/to/model_checkpoints/bert-base-uncased/"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = ProtoNetWrapper(model_name=base_model_path, wandb_run=None)
ckpt_path = "path/to/trained_checkpoints/proto_bert_base/199.ckpt"
model.load(ckpt_path, strict=True)

model.rebuild_memory_from_users(simulation_user_data, tokenizer)

eval_tasks = build_tasks(user_dataset = human_user_data, tokenizer = tokenizer, 
                        support_ratio =  30, task_prefix = "test") 

f1, acc, bacc = evaluate_episode_userproto(
                eval_tasks, model, mix_neighbors=True,
                print_report = True
            )
print(f1)