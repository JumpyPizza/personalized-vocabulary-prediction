import torch
import json
import random
from tqdm import tqdm
import sys
sys.path.append("path/to/repository/meta_learn")
from bert_model import BertTokenTask



def prepare_data(data, tokenizer, 
                 prediction_model, # the model to provide reward, now it's a reptile based model 
                 num_initial_samples, 
                 num_candidate_samples,
                 ):
   

        # initialize different initial samples for each user 
        single_user_dataloader = BertTokenTask(data, tokenizer, support_ratio=num_initial_samples)

        initial_support_input, _, support_words = single_user_dataloader.get_data(support_batch_size=num_initial_samples, query_batch_size=0, ensure_non_zero=True)
        initial_sample_preds, initial_sample_ground_labels, initial_sample_representations = prediction_model.evaluate_prediction(initial_support_input, return_representation=True)
        initial_ids = [word_to_idx[word] for word in support_words]
      
        initial_sample_representations = torch.tensor(initial_sample_representations)
        initial_sample_ground_labels = torch.tensor(initial_sample_ground_labels)
        initial_sample_preds = torch.tensor(initial_sample_preds)
        
        baseline_samples = single_user_dataloader.sample_query(random_sample=False, num_samples= num_candidate_samples)
        baseline_inputs = single_user_dataloader.get_data_by_samples(baseline_samples)

        return single_user_dataloader, \
               initial_support_input, initial_sample_preds, \
               initial_sample_ground_labels, initial_sample_representations, initial_ids, \
               baseline_inputs, baseline_samples


word_list = []
with open("path/to/data/word_list.txt", "r") as f:
    for line in f.readlines():
        if line.strip():    
            word_list.append(line.strip())
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
pool_length = len(word_list)


# first, load the simulation data, keys are user names
simulation_data = []
simulation_user_data = [] # data-only, no user names (sentence, tokens, labels)
cnt = 0
with open("path/to/generated_outputs/coca_simulation_llama/sentence_level_sim_labels_per_user.json", "r") as f:
    for line in tqdm(f.readlines()):
        line_data = json.loads(line)
        simulation_data.append(line_data)
        cnt += 1
        # if cnt == 3:
        #     break
user_names = [list(u.keys())[0] for u in simulation_data]

for name, data in zip(user_names, simulation_data):
    simulation_user_data.append(data[name]) # one element is a list of data
print(f"simulation user data: {len(simulation_user_data)}")       
random.shuffle(simulation_user_data)


LOAD_MODEL_PATH = "path/to/repository/meta_learn/ckpt/reptile_simulation_pretrain.pt"
