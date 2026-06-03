# Deprecated
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from bert_model import  BertTokenTask, BertTokenClf
from tqdm import tqdm
import json
import random
from maml_model import MAMLWrapper
import torch 
from transformers import AutoTokenizer
import wandb
from sklearn.metrics import f1_score, balanced_accuracy_score
import numpy as np 

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)



def train(model, tokenizer, first_order, train_user_data, inner_lr, outer_lr, inner_steps, epoch, wandb_run):
    
    

    
    
    maml = MAMLWrapper(model, inner_lr, outer_lr, inner_steps, first_order, wandb_run)
    

    for idx in range(epoch):
        episode = []
        for data in train_user_data:
            episode.append(BertTokenTask(data, tokenizer, support_ratio=50))
            break


        maml.meta_update(episode, support_batch_size=50, query_batch_size=500, max_query_subbatch_size=32, max_support_subbatch_size=32)



        if idx % 5 == 0:
            average_f1 = []
            average_balanced_acc = []
            for task in tqdm(episode, desc="evaluating"):
                evaluation_set = task.get_evaluation_set(task.query_set)
            
                res, labels = maml.evaluate(evaluation_set, 100)
                task_f1 =  f1_score(res, labels, average="macro")
                task_balanced_acc = balanced_accuracy_score(res, labels)
                average_f1.append(task_f1)
                average_balanced_acc.append(task_balanced_acc)
            average_f1_epoch = np.mean(average_f1)
            average_balanced_acc_epoch = np.mean(average_balanced_acc)
            if wandb_run is not None:
                wandb_run.log({"average_f1": average_f1_epoch, "average_balanced_acc": average_balanced_acc_epoch})
            else:
                print(f"average_f1: {average_f1_epoch}, average_balanced_acc: {average_balanced_acc_epoch}")
    

if __name__ == "__main__":
    pass

    # human_data = []
    # with open("path/to/generated_outputs/coca_simulation/sentence_level_user_labels.json", "r") as f:
    #     for line in tqdm(f.readlines()):
    #         data = json.loads(line)
    #         for key in data.keys():
    #             if "user" in key:
    #                 human_data.append(data[key])
    #                 break
    # print(len(human_data))
    # with open("path/to/generated_outputs/coca_simulation/sentence_level_real_user_labels.json", "w") as f:
    #     for data in human_data:
    #         f.write(json.dumps(data) + "\n")

    # user_data = []
    # cnt = 0
    # with open("path/to/generated_outputs/coca_simulation/sentence_level_user_labels.json", "r") as f:
    #     for line in tqdm(f.readlines()):
    #         user_data.append(json.loads(line))
    #         cnt += 1
    #         if cnt == 1:
    #             break

    # user_names = [list(u.keys())[0] for u in user_data]
    # single_user_data = []
    # for name, data in zip(user_names, user_data):
    #     single_user_data.append(data[name])

    # sampled_single_user_data = random.Random(0).sample(single_user_data, 200) # randomly sample some user tasks
 
    # first_order = True
    # inner_lr = 1e-5
    # outer_lr = 1e-4
    # epoch = 300
    # inner_steps = 5
    # # wandb_run = wandb.init(project="maml-test", name="maml-token-clf")
    # wandb_run = None

    # model = BertTokenClf("bert-base-uncased", 2)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # human_user_data = []
    # with open("path/to/generated_outputs/coca_simulation/sentence_level_real_user_labels.json", "r") as f:
    #     for line in tqdm(f.readlines()):
    #         human_user_data.append(json.loads(line))
    # print(f"human user data: {len(human_user_data)}")
    # train_user_data = human_user_data

    # train(model, tokenizer, first_order,train_user_data, inner_lr, outer_lr, inner_steps, epoch, wandb_run)
