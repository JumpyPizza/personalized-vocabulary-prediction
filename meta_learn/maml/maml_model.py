# Deprecated 

import torch
import torch.nn as nn
import higher
from tqdm import tqdm
import wandb
import numpy as np


class MAMLWrapper():
    def __init__(self, 
                 model: nn.Module,
                 inner_lr: float,
                 outer_lr: float,
                 inner_steps: int,
                 first_order: bool,
                 wandb_run):
        if not torch.cuda.is_available():
            raise ValueError("No GPU found")
        self.model = model.to('cuda')
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)

        self.wandb_run = wandb_run

    def meta_update(self, tasks, support_batch_size, query_batch_size, max_query_subbatch_size=32, max_support_subbatch_size=20):
        self.model.train()
        meta_loss = 0.0
        track_higher_grads = not self.first_order
        self.meta_optimizer.zero_grad()
        for task in tqdm(tasks, desc="training on tasks"):
            support_input, query_input = task.get_data(support_batch_size, query_batch_size)
  
            # Set up inner-loop optimizer
            inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

            # Higher's inner loop context manager
            with higher.innerloop_ctx(
                self.model,
                inner_optimizer,
                copy_initial_weights=False,
                track_higher_grads=track_higher_grads
            ) as (fmodel, diffopt):

                # Step 1: Inner loop adaptation on support set
                support_loss = self._adapt_to_task(fmodel, diffopt, support_input, max_support_subbatch_size)

                # Step 2: Evaluate on query set and accumulate gradients
                task_loss = self._evaluate_and_accumulate(fmodel, query_input, max_query_subbatch_size)
                # Log losses to wandb
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "support_loss": support_loss,
                        "task_loss": task_loss,
                    })

                meta_loss += task_loss

        # Step 3: Meta optimization step
        meta_loss /= len(tasks)
        if self.wandb_run is not None:
            self.wandb_run.log({
                "meta_loss": meta_loss
            })
        

        # Normalize gradients across tasks
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad /= len(tasks)

        self.meta_optimizer.step()
        return meta_loss

    def _adapt_to_task(self, fmodel, diffopt, support_input, max_subbatch_size):
        average_support_loss = 0.0
        input_ids = support_input["input_ids"]
        attention_mask = support_input["attention_mask"]
        labels = support_input["labels"]
        batch_size = input_ids.shape[0]
        for _ in range(self.inner_steps):
            mini_batch_support_loss = 0.0
            for i in range(0, batch_size, max_subbatch_size):
                end = min(i + max_subbatch_size, batch_size)
                sub_input = {
                    "input_ids": input_ids[i:end],
                    "attention_mask": attention_mask[i:end],
                    "labels": labels[i:end]
                }
                support_output = fmodel(sub_input)
                mini_batch_support_loss += support_output.loss
            average_support_loss += mini_batch_support_loss.item()
            diffopt.step(mini_batch_support_loss)
        average_support_loss /= self.inner_steps
        return average_support_loss

    def _evaluate_and_accumulate(self, fmodel, query_input, max_subbatch_size):
        input_ids = query_input["input_ids"]
        attention_mask = query_input["attention_mask"]
        labels = query_input["labels"]
        batch_size = input_ids.shape[0]

        total_loss = 0.0


        for i in range(0, batch_size, max_subbatch_size):
            end = min(i + max_subbatch_size, batch_size)

            sub_input = {
                "input_ids": input_ids[i:end],
                "attention_mask": attention_mask[i:end],
                "labels": labels[i:end]
            }

            query_output = fmodel(sub_input)
            query_loss = query_output.loss
            total_loss += query_loss
            # query_loss.backward()
            
        total_loss.backward()
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
                    "input_ids": evaluate_set["input_ids"][i:end],
                    "attention_mask": evaluate_set["attention_mask"][i:end],
                    "labels": evaluate_set["labels"][i:end]
                }
                
                output, labels = self.model.evaluate_prediction(sub_input)
                evaluation_outputs.extend(output)
                evaluation_labels.extend(labels)
        evaluation_outputs = np.vstack(evaluation_outputs)
        evaluation_labels = np.vstack(evaluation_labels)
        return evaluation_outputs, evaluation_labels
    
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    # def meta_update(self, 
    #              tasks, support_batch_size, query_batch_size):
    #     self.model.train()
    #     meta_loss = 0.0
    #     track_higher_grads = not self.first_order
    #     for task in tqdm(tasks, desc="training on tasks"):
    #         support_input, query_input = task.get_data(support_batch_size, query_batch_size)
    #         inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
    #         with higher.innerloop_ctx(self.model, inner_optimizer, copy_initial_weights=False, track_higher_grads=track_higher_grads) as (fmodel, diffopt):
    #             for _ in range(self.inner_steps):
    #                 support_output = fmodel(support_input)
    #                 support_loss = support_output.loss
    #                 diffopt.step(support_loss)

    #             query_output = fmodel(query_input)
    #             task_loss = query_output.loss
    #             self.loss_tracker.append(task_loss.item())
    #             meta_loss += task_loss
    #     meta_loss /= len(tasks)
    #     self.meta_optimizer.zero_grad()
    #     meta_loss.backward()
    #     self.meta_optimizer.step()
    #     return meta_loss.item()