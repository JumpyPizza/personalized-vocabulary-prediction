import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import numpy as np
from copy import deepcopy

class ReptileWrapper():
    def __init__(self, 
                 model: nn.Module,
                 inner_lr: float,
                 outer_lr: float,
                 inner_steps: int,
                 wandb_run):
        if not torch.cuda.is_available():
            raise ValueError("No GPU found")
        if torch.cuda.device_count() > 1:
            self.model = model.to('cuda:0')
        else:
            self.model = model.to('cuda')
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        # Meta optimizer (applied to the meta model parameters)
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)
        self.task_loss_tracker = []
        self.support_loss_tracker = []
        self.wandb_run = wandb_run
    
    def meta_update(self, tasks, support_batch_size, query_batch_size, 
                    max_query_subbatch_size=32, max_support_subbatch_size=20):
        self.model.train()
        # meta_loss = 0.0
        num_tasks = len(tasks)
        meta_device = self.model.device
        # Create an accumulator for the parameter differences
        delta_acc = { key: torch.zeros_like(param, device=meta_device) 
                      for key, param in self.model.state_dict().items() }

        self.meta_optimizer.zero_grad()
        episode_support_loss = []
        for task in tasks: # each iteration 
       
       
            support_input, query_input, _ = task.get_data(support_batch_size, query_batch_size)
       
        
            # Save a copy of the meta-model's current parameters (baseline)
            meta_weights = { key: param.clone().detach()
                             for key, param in self.model.state_dict().items() }
            
            # Create a local copy of the model for inner-loop adaptation.
            # Note: We use deepcopy to make sure that inner updates do not affect the meta-model.
            local_model = deepcopy(self.model) # local model does not copy DP but only module
            local_model.train() 
            if torch.cuda.device_count() > 1:
                local_model = local_model.to('cuda:1')
            else:
                local_model = local_model.to('cuda')
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=self.inner_lr)
            
            # --- Inner Loop Adaptation on Support Set ---
            support_loss = self._adapt_to_task(local_model, local_optimizer, support_input, max_support_subbatch_size)
            episode_support_loss.append(support_loss)
            # # --- Optional Query Evaluation (for logging) ---
            # task_loss = self._evaluate_without_backprop(local_model, query_input, max_query_subbatch_size)
            # self.wandb_run.log({
            #     "support_loss": support_loss,
            # })
            # --- Accumulate Parameter Differences ---
            local_state = local_model.state_dict()
            for key in delta_acc:
                # Compute the difference between the task-adapted parameters and the meta baseline.
                delta_acc[key] += (local_state[key].to(meta_device) - meta_weights[key])
        
      
        
        # --- Meta Update ---
        # For each meta-parameter, set its gradient as the negative average of the accumulated differences.
        # This makes the update: θ = θ - outer_lr * (-avg_delta) = θ + outer_lr * avg_delta.
        for name, param in self.model.named_parameters():
            grad = - delta_acc[name] / num_tasks
            # Ensure the gradient is on the same device as the parameter.
            param.grad = grad.to(param.device)
        
        self.meta_optimizer.step() # update after each iteration 
        return episode_support_loss
    
    def meta_step(self, support_input_list, max_support_subbatch_size=20):
        self.model.train()
        # meta_loss = 0.0
   
        meta_device = self.model.device
        # Create an accumulator for the parameter differences
        delta_acc = { key: torch.zeros_like(param, device=meta_device) 
                      for key, param in self.model.state_dict().items() }

        self.meta_optimizer.zero_grad()
        episode_support_loss = []
        for support_input in support_input_list:
        # Save a copy of the meta-model's current parameters (baseline)
            meta_weights = { key: param.clone().detach()
                                for key, param in self.model.state_dict().items() }
            
            # Create a local copy of the model for inner-loop adaptation.
            # Note: We use deepcopy to make sure that inner updates do not affect the meta-model.
            local_model = deepcopy(self.model) # local model does not copy DP but only module
            local_model.train() 
            if torch.cuda.device_count() > 1:
                local_model = local_model.to('cuda:1')
            else:
                local_model = local_model.to('cuda')
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=self.inner_lr)
            
            # --- Inner Loop Adaptation on Support Set ---
            support_loss = self._adapt_to_task(local_model, local_optimizer, support_input, max_support_subbatch_size)
            episode_support_loss.append(support_loss)
            # # ---  Query Evaluation for logging ---
            # task_loss = self._evaluate_without_backprop(local_model, query_input, max_query_subbatch_size)
            # self.wandb_run.log({
            #     "support_loss": support_loss,
            # })
            # --- Accumulate Parameter Differences ---
            local_state = local_model.state_dict()
            for key in delta_acc:
                # Compute the difference between the task-adapted parameters and the meta baseline.
                delta_acc[key] += (local_state[key].to(meta_device) - meta_weights[key])
        
      
        
        # --- Meta Update ---
        # For each meta-parameter, set its gradient as the negative average of the accumulated differences.
        # This makes the update: θ = θ - outer_lr * (-avg_delta) = θ + outer_lr * avg_delta.
        for name, param in self.model.named_parameters():
            grad = - delta_acc[name] 
            # Ensure the gradient is on the same device as the parameter.
            param.grad = grad.to(param.device)
        
        self.meta_optimizer.step() # update after each iteration 
        return episode_support_loss

    def _adapt_to_task(self, local_model, local_optimizer, support_input, max_subbatch_size):
        average_support_loss = 0.0
        # Unpack support inputs
        input_ids = support_input["input_ids"]
        attention_mask = support_input["attention_mask"]
        labels = support_input["labels"]
        batch_size = input_ids.shape[0]
        
        for _ in range(self.inner_steps):
            mini_batch_support_loss = 0.0
            for i in range(0, batch_size, max_subbatch_size):
                end = min(i + max_subbatch_size, batch_size)
                sub_input = {
                    "input_ids": input_ids[i:end].to(local_model.device),
                    "attention_mask": attention_mask[i:end].to(local_model.device),
                    "labels": labels[i:end].to(local_model.device)
                }
                support_output = local_model(sub_input)
                mini_batch_support_loss += support_output.loss
            average_support_loss += mini_batch_support_loss.item()
            local_optimizer.zero_grad()
            mini_batch_support_loss.backward()
            local_optimizer.step()
        average_support_loss /= self.inner_steps
        return average_support_loss

    def _evaluate_without_backprop(self, local_model, query_input, max_subbatch_size):
        local_model.eval()
        input_ids = query_input["input_ids"]
        attention_mask = query_input["attention_mask"]
        labels = query_input["labels"]
        batch_size = input_ids.shape[0]
        total_loss = 0.0
        with torch.no_grad():
            for i in range(0, batch_size, max_subbatch_size):
                end = min(i + max_subbatch_size, batch_size)
                sub_input = {
                    "input_ids": input_ids[i:end].to(local_model.device),
                    "attention_mask": attention_mask[i:end].to(local_model.device),
                    "labels": labels[i:end].to(local_model.device)
                }
                query_output = local_model(sub_input)
                total_loss += query_output.loss
        return total_loss.item()

    def evaluate(self, evaluate_set, evaluate_batch_size):
        self.model.eval()
        evaluation_outputs = []
        evaluation_labels = []
        evaluate_size = len(evaluate_set["input_ids"])
        with torch.no_grad():
            for i in range(0, evaluate_size, evaluate_batch_size):
                end = min(i + evaluate_batch_size, evaluate_size)
                sub_input = {
                    "input_ids": evaluate_set["input_ids"][i:end].to(self.model.device),
                    "attention_mask": evaluate_set["attention_mask"][i:end].to(self.model.device),
                    "labels": evaluate_set["labels"][i:end].to(self.model.device)
                }
                output, labels = self.model.evaluate_prediction(sub_input)
                evaluation_outputs.extend(output)
                evaluation_labels.extend(labels)
        evaluation_outputs = np.vstack(evaluation_outputs)
        evaluation_labels = np.vstack(evaluation_labels)
        return evaluation_outputs, evaluation_labels
    
    def get_logits(self, evaluate_set, evaluate_batch_size):
        self.model.eval()
        evaluation_outputs = []
        evaluate_size = len(evaluate_set["input_ids"])
        with torch.no_grad():
            for i in range(0, evaluate_size, evaluate_batch_size):
                end = min(i + evaluate_batch_size, evaluate_size)
                sub_input = {
                    "input_ids": evaluate_set["input_ids"][i:end].to(self.model.device),
                    "attention_mask": evaluate_set["attention_mask"][i:end].to(self.model.device),
                    "labels": evaluate_set["labels"][i:end].to(self.model.device)
                }
                _, _, logits = self.model.evaluate_prediction(sub_input, return_logits=True) #
                
                evaluation_outputs.extend(logits) # bs, num_labels

        evaluation_outputs = np.vstack(evaluation_outputs)
   
        return evaluation_outputs

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
    
        self.model.load_state_dict(torch.load(path))
