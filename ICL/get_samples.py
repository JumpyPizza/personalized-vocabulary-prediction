"""
Run meta-learning on simulation samples, get all low_conf samples;
for each learner:
    - initial words: List 
    - sample words: Dict[word: [true, pred, conf]
"""

import sys 
sys.path.append("../llm_inference")
sys.path.append("../meta_learn")
sys.path.append("../AL")

from bert_model import BertTokenTask, BertTokenClf 
from train_reptile import finetune_and_evaluate
from active_llm import UserSimulationOracle

from transformers import AutoTokenizer 
import torch 
from tqdm import tqdm
import json



model_path = "path/to/model_checkpoints"
data_path = "path/to/data"

word_list_path = "./word_list.txt"
bert_base_uncased_path = model_path + "/bert-base-uncased"

# user:data
# user_file_path = r"G:\data\server_data\vocab_prediction\output\coca_simulation_llama\sentence_level_sim_labels_per_user.json"
user_file_path = data_path + "/sentence_level_real_user_labels.json"
LOAD_MODEL_PATH = model_path + "/reptile_simulation_pretrain.pt"

with open(word_list_path, "r", encoding="utf-8") as f: 
    word_list = [line.strip().lower() for line in f.readlines()]
    

def get_inconfident_samples(prediction_model, batch_size, 
                            eval_data_loader, data_loader,     
                            initial_words, 
                            word_list, threshold):
     
    
    finetune_samples = [data_loader.get_sample_by_word(w) for w in initial_words]
    inputs = data_loader.get_data_by_samples(finetune_samples)

    finetune_and_evaluate(prediction_model,  eval_data_loader, inputs, 
                                inner_lr = 1e-5, outer_lr= 1e-4, inner_steps=5,                            
                                train_iter_num = 2, evaluate = False)
  
    samples = []
    sample_words = []
    for w in word_list:
        sample = data_loader.get_sample_by_word(w)
        if sample is not None:
            samples.append(sample)
            sample_words.append(w)
    inputs = data_loader.get_data_by_samples(samples)
    pred_labels, true_labels, pred_logits, indices = prediction_model.evaluate_prediction(
                inputs, batch_size = batch_size, return_logits=True
            )
    inconfident_errors = []
    for pred, true, logits, idx in zip(pred_labels, true_labels, pred_logits, indices):

        prob = torch.softmax(torch.tensor(logits), dim=-1)
        if prob[pred] < threshold:
            if int(pred) != int(true):
                word = sample_words[idx]
                sample = {
                    word : [int(true), int(pred), float(prob[pred])]
                }
                inconfident_errors.append(sample)
                
    return inconfident_errors
                

if __name__ == "__main__":
    
    ########### load sim user data ###################
    # sim_user_data = []
    # with open(user_file_path, "r") as f:
        
    #     for idx, line in enumerate(tqdm(f.readlines())):
    #         sim_user_data.append(json.loads(line))
    #         break 
    # user_data = sim_user_data[0]
    # user_data = user_data[list(user_data.keys())[0]]
    
    human_user_data = []
    with open(user_file_path, "r") as f:
        
        for idx, line in enumerate(tqdm(f.readlines())):
            human_user_data.append(json.loads(line))

        
    print(f"human user data size: {len(human_user_data)}")
    
    prediction_model = BertTokenClf(bert_base_uncased_path, 2).cuda()
    tokenizer = AutoTokenizer.from_pretrained(bert_base_uncased_path)
    
    for idx, user_data in enumerate(human_user_data):
        if idx > 10:
            print(f" === user {idx} ==== ")
            all_initial_samples = []
            prediction_model.load_state_dict(torch.load(LOAD_MODEL_PATH))
            evaluation_loader = BertTokenTask(user_data, tokenizer, support_ratio=30)
            data_loader = BertTokenTask(user_data, tokenizer, support_ratio=0.99)
        
            user_oracle = UserSimulationOracle(user_data=user_data, tokenizer=tokenizer, initial_sample_num=30)
            while True:
                try:
                    initial_words_labels = user_oracle.query_initial_samples()
                    break
                except AssertionError:
                    continue
            initial_words = initial_words_labels[0]
            initial_labels = initial_words_labels[1]
            errors = get_inconfident_samples(prediction_model = prediction_model,
                                            batch_size=500, 
                                    eval_data_loader = evaluation_loader, 
                                    data_loader = data_loader,     
                                    initial_words = initial_words, 
                                    word_list = word_list, 
                                    threshold = 0.55)
            print(len(errors))
            all_initial_samples.append(initial_words_labels)
            for _ in range(20):
                try:
                    initial_words_labels = user_oracle.query_initial_samples() #(word, label)
                
                    all_initial_samples.append(initial_words_labels)
                except AssertionError:
                    continue
            user_sample = {
                "initial_samples":all_initial_samples,
                "errors":errors
            }
            with open(f"{idx}.json", "w", encoding="utf-8") as f:
                json.dump(user_sample, f)
