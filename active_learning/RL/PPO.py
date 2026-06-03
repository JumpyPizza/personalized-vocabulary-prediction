import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("path/to/repository/meta_learn")
from policy_model import TransformerPolicySequential
import torch
import torch.nn.functional as F
import json
import random
from tqdm import tqdm
import wandb
from bert_model import BertTokenTask, BertTokenClf
from train_reptile import finetune_and_evaluate
from transformers import AutoTokenizer

from data import prepare_data, LOAD_MODEL_PATH, pool_length, word_to_idx, idx_to_word, simulation_user_data




def rollout_one(init_inputs, prediction_model,
                policy, dataloader,
                init_states,    # (reprs, true_labels, pred_labels, initial_ids)
                inner_lr, outer_lr, inner_steps, K = 30):
    """
    Do one *sequential* rollout of length K,
    returning lists of (logps, values, actions) and final reward.
    """
    finetune_and_evaluate(
        prediction_model, dataloader,
        init_inputs, inner_lr, outer_lr, inner_steps,
        train_iter_num=5
    )
    reprs, y_true, y_pred, init_ids = init_states
    device = reprs.device

    logps, values, actions = [], [], []
    states_cache = []

    # at step 0, “current” state is empty, only use initial states to choose next 
    cur_state = torch.empty(0, reprs.size(1), device=device)
    cur_true  = torch.empty(0, dtype=torch.long, device=device)
    cur_pred  = torch.empty(0, dtype=torch.long, device=device)
    with torch.no_grad():
    # start rollout 
    # sequentially pick K candidates
        for t in range(K):
            logits, value = policy(
                reprs, y_true, y_pred,
                cur_state, cur_true, cur_pred
            )  # logits:[n_candidates], value:scalar

            # sample one action (you could also sample top-K by repeating this)
            dist = torch.distributions.Categorical(logits=logits)
            action   = dist.sample()     
            logp     = dist.log_prob(action)          # scalar 
            action = action.item()
            logps.append(logp)
            values.append(value.squeeze(0))
            actions.append(action)
            states_cache.append((cur_state, cur_true, cur_pred))
            # fetch that sample’s embedding & labels from dataloader
            sample = dataloader.get_sample_by_word(idx_to_word[action])

            if sample is None:
                # if we happen to pick an invalid action, stop early
                # sometimes idx_to_word returns None 
                break
            sample_input = dataloader.get_data_by_samples([sample])
            
            y_new_pred, y_new_true, x_new = prediction_model.evaluate_prediction(sample_input, return_representation = True)
            y_new_pred = torch.tensor(y_new_pred, device=device)
            y_new_true = torch.tensor(y_new_true, device=device)
            x_new = torch.tensor(x_new, device=device)
            # append to the “current” context for next step
            cur_state = torch.cat([cur_state, x_new], dim=0)
            cur_true  = torch.cat([cur_true,  y_new_true], dim=0)
            cur_pred  = torch.cat([cur_pred,  y_new_pred], dim=0)

    # once we have our trajectory of actions, build the sampled_inputs
    picked_samples = [dataloader.get_sample_by_word(idx_to_word[a]) for a in actions]
    picked_samples = [s for s in picked_samples if s is not None]
    sampled_inputs = dataloader.get_data_by_samples(picked_samples)

    # compute reward via F1 improvement
    before_f1, after_f1 = finetune_and_evaluate(
        prediction_model, dataloader,
        sampled_inputs, inner_lr, outer_lr, inner_steps,
        train_iter_num=5
    )
    R = (after_f1 - before_f1) * 100.0

    prediction_model.load_state_dict(torch.load(LOAD_MODEL_PATH))
    return {
            'init_states': init_states,
            'logps': logps,
            'values': values,
            'actions': actions,
            'states': states_cache,
            'R_final': R
            }


def compute_gae(values, R_final, gamma=0.99, lam=0.95):
    """
        Given list of values [V_0, …, V_{T-1}], and final reward R_final at step T-1,
        compute returns and GAE advantages.
    """
    advantages = []
    gae = 0
    T = len(values)
    rewards = [0.0] * (T-1) + [R_final] # reward sequence: 0, 0, 0 ... R_final
    values = values + [0.0]  # append a zero to V for terminal next-value 

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t] # TD
        gae = delta + gamma * lam * gae # gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
     # normalize
    advantages = torch.tensor(advantages)
    returns = torch.tensor(returns)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return advantages, returns

def ppo_update(policy, optimizer, trajectories, ppo_epochs, ppo_clip_eps, critic_coef, entropy_coef):
    """
    trajectories: list of dicts {'init_states': [reprs, y_true, y_pred, init_ids],
                                 'logps':   [T],
                                 'values': [T],
                                 'actions':[T],
                                 'states': [(cur_state, cur_true, cur_pred)*T],
                                 'R':       scalar}
    """
    # 1) Build flat tensors of logps, values, rewards, advantages
    all_logps = torch.cat([torch.stack(tr['logps'])    for tr in trajectories])
    # all_vals  = torch.cat([torch.stack(tr['values'])  for tr in trajectories])
    # all_actions = torch.cat([torch.tensor(tr['actions']) for tr in trajectories]).to(all_logps.device) # action is an idx choice
    
    returns, advantages = [], []
    for tr in trajectories:
        ret, adv = compute_gae(tr['values'], tr['R_final'])
        returns.append(ret)
        advantages.append(adv)
    all_returns = torch.cat(returns).to(all_logps.device) # for all trajectories
    all_advantages = torch.cat(advantages).to(all_logps.device)
    
    for _ in range(ppo_epochs):
        # 2) Compute the loss
        new_logps, new_values, entropies = [], [], []

        # for each rollout, replay the trajectory
        for tr in trajectories:
            reprs, y_true, y_pred, _ = tr['init_states']
            for (cur_x, cur_true, cur_pred), a in zip(tr['states'], tr['actions']):
                logits, v = policy(
                    reprs, y_true, y_pred,
                    cur_x, cur_true, cur_pred
                )
                a = torch.tensor(a, device=logits.device)
                dist = torch.distributions.Categorical(logits=logits)
                new_logps.append(dist.log_prob(a))
                new_values.append(v.squeeze(0))
                entropies.append(dist.entropy())

        new_logps = torch.stack(new_logps)
        new_values = torch.stack(new_values)
        entropy = torch.stack(entropies).mean()

        approx_kl = 0.5 * ((new_logps - all_logps) ** 2).mean().item()
        

        ratio = torch.exp(new_logps - all_logps)
        surr1 = ratio * all_advantages
        surr2 = torch.clamp(ratio, 1-ppo_clip_eps, 1+ppo_clip_eps) * all_advantages
        actor_loss  = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(new_values, all_returns)
        loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        return approx_kl, actor_loss.item(), critic_loss.item()
    

        
if __name__ == "__main__":

    

    wandb_run = wandb.init(project="vocab-prediction-sampler-rl", name="ppo_seq")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", max_length=512)
    prediction_model = BertTokenClf("bert-base-uncased", 2).to("cuda")
    prediction_model.load_state_dict(torch.load(LOAD_MODEL_PATH))

    num_initial_samples = 20 
    num_candidate_samples = 30 

    inner_lr = 1e-5
    outer_lr = 1e-4
    inner_steps = 5

    policy = TransformerPolicySequential(
        d_emb=768,
        n_candidates=pool_length,
        n_heads=4, n_layers=4,
        temperature=1.0
    ).cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
    train_epochs = 100
    init_time = 5 # how many different initial samples to use for each user in each epoch
    rollout_iter = 5 # each update uses this many rollouts

    ppo_epochs = 5 # how many epochs to perform on the same rollouts 
    mini_epochs = 5 # how many iter to sample different rollouts
    ppo_clip_eps = 0.2
    critic_coef = 0.5
    entropy_coef = 0.01
    
    for epoch in range(train_epochs):                             
        for data in simulation_user_data:                
            for _ in range(init_time):    #init_time initial samples chosen for each user, each epoch 
                # build initial support / baseline
                (dataloader,
                init_inputs, init_preds,
                init_trues, init_reprs, init_ids,
                _, _) = prepare_data(
                    data, tokenizer,
                    prediction_model,
                    num_initial_samples,
                    num_candidate_samples
                )
                init_states = (init_reprs.cuda(),
                            init_trues.cuda(),
                            init_preds.cuda(),
                            init_ids)

                
                for _ in range(mini_epochs):
                    # collect rollouts
                    trajectories = []
                    tr_avg_R = []
                    for _ in range(rollout_iter):
                        try:
                            tr = rollout_one(
                                init_inputs, prediction_model,
                                policy, dataloader,
                                init_states,
                                inner_lr, outer_lr, inner_steps
                            )
                            R_final = tr['R_final']
                            tr_avg_R.append(R_final)
                            trajectories.append(tr)
                        except Exception as e:
                            print(f"Error in rollout: {e}")
                            continue
                    tr_avg_R = sum(tr_avg_R) / rollout_iter # average reward of these rollouts

                    # PPO update on the collected rollouts
                  
                    approx_kl, actor_loss, critic_loss = ppo_update(policy, optimizer, trajectories, ppo_epochs, ppo_clip_eps, critic_coef, entropy_coef)
              
                    # print(f"avg R: {tr_avg_R}, kl: {approx_kl}, actor_loss: {actor_loss}, critic_loss: {critic_loss}")
                    wandb_run.log({
                        "avg R": tr_avg_R,
                        "kl": approx_kl,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss
                    })