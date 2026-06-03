import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
from tqdm import tqdm
import sys
sys.path.append("path/to/repository/meta_learn")
from bert_model import BertTokenTask, BertTokenClf
from train_reptile import finetune_and_evaluate
from transformers import AutoTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from data import prepare_data, word_to_idx, idx_to_word, pool_length
from policy_model import ReinforceOneShot, ActorCriticOneShot
import wandb












# finetune_and_evaluate(prediction_model,  single_user_dataloader, sampled_inputs,
#                         inner_lr, outer_lr, inner_steps,                                  
#                         train_iter_num = 5)
# prediction_model.load_state_dict(torch.load(load_model_path))

# each episode has some batches 
# each batch is from the same user 

      

def actor_critic_step(policy_model, optimizer, prediction_model, 
                      single_user_dataloader, R_baseline,
                      initial_samples, initial_sample_labels, initial_sample_preds, initial_ids,
                      inner_lr, outer_lr, inner_steps, beta):

        logits, value = policy_model(initial_samples, initial_sample_labels, initial_sample_preds)
        idxs, logps     = policy_model.sample_subset(logits, initial_ids)
        idxs = idxs.tolist()
        policy_sampled_words = [idx_to_word[idx] for idx in idxs]

        policy_sampled_samples = [single_user_dataloader.get_sample_by_word(word) for word in policy_sampled_words]
        policy_sampled_samples = [s for s in policy_sampled_samples if s is not None]

        sampled_inputs = single_user_dataloader.get_data_by_samples(policy_sampled_samples)
        
        
       
        before_sample_f1, after_sample_f1 = finetune_and_evaluate(prediction_model,  single_user_dataloader, sampled_inputs,
                        inner_lr, outer_lr, inner_steps,                                  
                        train_iter_num = 5)
        R = (after_sample_f1 - before_sample_f1) * 100
        R_t = torch.tensor(R, device="cuda", dtype=torch.float32)

        A = R_t - R_baseline 

        # actor loss
        loss_pg = - logps.sum() * A
        ent     = -(F.softmax(logits,0) * F.log_softmax(logits,0)).sum() # entropy bonus to encourage exploration
        loss_a  = loss_pg - beta * ent
       
        # critic loss
        loss_c  = F.mse_loss(value, A)

        optimizer.zero_grad()
        (loss_a + loss_c).backward()
        nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        return loss_a.item(), loss_c.item(), R, A.item()


def reinforce_step(policy_model, optimizer, prediction_model, 
                      single_user_dataloader, R_baseline,
                      initial_samples, initial_sample_labels, initial_sample_preds, initial_ids,
                      inner_lr, outer_lr, inner_steps, beta):

        logits = policy_model(initial_samples, initial_sample_labels, initial_sample_preds)
        idxs, logps     = policy_model.sample_subset(logits, initial_ids)
        idxs = idxs.tolist()
        policy_sampled_words = [idx_to_word[idx] for idx in idxs]

        policy_sampled_samples = [single_user_dataloader.get_sample_by_word(word) for word in policy_sampled_words]
        policy_sampled_samples = [s for s in policy_sampled_samples if s is not None]

        sampled_inputs = single_user_dataloader.get_data_by_samples(policy_sampled_samples)
        
        
       
        before_sample_f1, after_sample_f1 = finetune_and_evaluate(prediction_model,  single_user_dataloader, sampled_inputs,
                        inner_lr, outer_lr, inner_steps,                                  
                        train_iter_num = 5)
        R = (after_sample_f1 - before_sample_f1) * 100
        R_t = torch.tensor(R, device="cuda", dtype=torch.float32)

        A = R_t - R_baseline 

        # actor loss
        loss_pg = - logps.sum() * A
        ent     = -(F.softmax(logits,0) * F.log_softmax(logits,0)).sum() # entropy bonus to encourage exploration
        loss_a  = loss_pg - beta * ent
       
        optimizer.zero_grad()
        loss_a.backward()
        optimizer.step()
        return loss_a.item(), R, A.item()


if __name__ == "__main__":
    ####Actor Critic One Shot ####
    wandb_run = wandb.init(project="vocab-prediction-sampler-rl", name="actor-critic-one-shot")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", max_length=512)
    prediction_model = BertTokenClf("bert-base-uncased", 2).cuda()
    load_model_path = "path/to/repository/meta_learn/ckpt/reptile_simulation_pretrain.pt"
    # print(f"loading model from {load_model_path}")


    random.Random(0).shuffle(simulation_user_data)
    # policy model hyperparameters
    num_initial_samples = 20
    num_candidate_samples = 30
    

    beta = 0.01   # entropy bonus weight

    # reptile hyperparameters
    inner_lr = 1e-5
    outer_lr = 1e-4
    inner_steps = 5
    
    policy_model_lr = 5e-4
    policy_model = ActorCriticOneShot(device="cuda", n_candidates=pool_length, K=num_candidate_samples)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=policy_model_lr)
    #  prediction_pretrain_model_path, # the pretrain ckpt for the prediction model 
    total_epochs = 10000
    iteration_num = 20
    print(f"total epochs: {total_epochs}")
    print(f"iteration num: {iteration_num}")
    print("training starts!")
    for epoch in range(total_epochs):
        for data in simulation_user_data: 
            prediction_model.load_state_dict(torch.load(load_model_path)) # restore prediction model states at each reward compute 
            single_user_dataloader, \
            initial_support_input, initial_sample_preds, \
            initial_sample_ground_labels, initial_sample_representations, initial_ids, \
            baseline_inputs, baseline_samples = prepare_data(data, tokenizer, prediction_model, num_initial_samples, num_candidate_samples)

            old_f1, finetuned_initial_f1 = finetune_and_evaluate(prediction_model,  single_user_dataloader, initial_support_input,
                            inner_lr, outer_lr, inner_steps,                                  
                            train_iter_num = 5)
            ############### save the initially finetuned model for later use ###############
            temp_dir = "./temp"
            os.makedirs(temp_dir, exist_ok=True)

            # Define the full path within the temp directory
            temp_initial_path = os.path.join(temp_dir, "temp_initial_model.pt")

            # Save the model
            torch.save(prediction_model.state_dict(), temp_initial_path)
            ##################################################################### 
            old_baseline_f1, finetuned_baseline_f1 = finetune_and_evaluate(prediction_model,  single_user_dataloader, baseline_inputs,
                            inner_lr, outer_lr, inner_steps,                                  
                            train_iter_num = 5)
            R_baseline = (finetuned_baseline_f1 - old_baseline_f1) * 100
            R_baseline_t = torch.tensor(R_baseline, device="cuda", dtype=torch.float32)
            ##################################################################### 
            for iter in range(iteration_num):
                prediction_model.load_state_dict(torch.load(temp_initial_path)) # restore prediction model states at each reward compute 
                loss_a, loss_c, R, A = actor_critic_step(policy_model, optimizer, prediction_model, 
                                single_user_dataloader, R_baseline_t,
                                initial_sample_representations, initial_sample_ground_labels, initial_sample_preds, initial_ids,
                                inner_lr, outer_lr, inner_steps, beta)
                # print(f"iter: {iter}, loss_a: {loss_a}, loss_c: {loss_c}, R: {R}, A: {A}")
                wandb_run.log({"loss_a": loss_a, "loss_c": loss_c, "R": R, "A": A})
    
    #### Reinforce One Shot ####

    wandb_run = wandb.init(project="vocab-prediction-sampler-rl", name="reinforce-one-shot")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", max_length=512)
    prediction_model = BertTokenClf("bert-base-uncased", 2).to("cuda")
    load_model_path = "path/to/repository/meta_learn/ckpt/reptile_simulation_pretrain.pt"
    # print(f"loading model from {load_model_path}")


    random.Random(0).shuffle(simulation_user_data)
    # policy model hyperparameters
    num_initial_samples = 20
    num_candidate_samples = 30
    

    beta = 0.01   # entropy bonus weight

    # reptile hyperparameters
    inner_lr = 1e-5
    outer_lr = 1e-4
    inner_steps = 5
    
    policy_model_lr = 5e-4
    policy_model = ReinforceOneShot(device="cuda", n_candidates=pool_length, K=num_candidate_samples)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=policy_model_lr)
    #  prediction_pretrain_model_path, # the pretrain ckpt for the prediction model 
    total_epochs = 10000
    iteration_num = 20
    print(f"total epochs: {total_epochs}")
    print(f"iteration num: {iteration_num}")
    print("training starts!")
    for epoch in range(total_epochs):
        for data in simulation_user_data: 
            prediction_model.load_state_dict(torch.load(load_model_path)) # restore prediction model states at each reward compute 
            single_user_dataloader, \
            initial_support_input, initial_sample_preds, \
            initial_sample_ground_labels, initial_sample_representations, initial_ids, \
            baseline_inputs, baseline_samples = prepare_data(data, tokenizer, prediction_model, num_initial_samples, num_candidate_samples)

            old_f1, finetuned_initial_f1 = finetune_and_evaluate(prediction_model,  single_user_dataloader, initial_support_input,
                            inner_lr, outer_lr, inner_steps,                                  
                            train_iter_num = 5)
            ############### save the initially finetuned model for later use ###############
            temp_dir = "./temp_2"
            os.makedirs(temp_dir, exist_ok=True)

            # Define the full path within the temp directory
            temp_initial_path = os.path.join(temp_dir, "temp_initial_model_2.pt")

            # Save the model
            torch.save(prediction_model.state_dict(), temp_initial_path)
            ##################################################################### 
            old_baseline_f1, finetuned_baseline_f1 = finetune_and_evaluate(prediction_model,  single_user_dataloader, baseline_inputs,
                            inner_lr, outer_lr, inner_steps,                                  
                            train_iter_num = 5)
            R_baseline = (finetuned_baseline_f1 - old_baseline_f1) * 100
            R_baseline_t = torch.tensor(R_baseline, device="cuda", dtype=torch.float32)
            ##################################################################### 
            for iter in range(iteration_num):
                prediction_model.load_state_dict(torch.load(temp_initial_path)) # restore prediction model states at each reward compute 
                loss_a, R, A = reinforce_step(policy_model, optimizer, prediction_model, 
                                single_user_dataloader, R_baseline_t,
                                initial_sample_representations, initial_sample_ground_labels, initial_sample_preds, initial_ids,
                                inner_lr, outer_lr, inner_steps, beta)
                # print(f"iter: {iter}, loss_a: {loss_a}, loss_c: {loss_c}, R: {R}, A: {A}")
                wandb_run.log({"loss_a": loss_a,  "R": R, "A": A})