
from reptile import ReptileWrapper
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from bert_model import BertTokenTask, BertTokenClf
import random
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report
import wandb
from copy import deepcopy
import argparse
from pathlib import Path

def train(model, tokenizer, test_user_data,
          iteration_num, task_num_in_episode, inner_lr, outer_lr, inner_steps_train, inner_steps_test,
          train_support_ratio, test_support_ratio,
          support_batch_size = 50, query_batch_size = 0, max_support_subbatch_size = 50, 
          wandb_run = None, use_simulation = False, simulation_user_data = None, save_model_path = None):
    """
    human user data will be used as test_user_data; 
    In each iteration;
        - sample a episode from train_user_data, which contains task_num_in_episode tasks
        - perform meta-learning on this episode 
    train_support_ratio: how many training samples in each task will be included; e.g., 0.1 will only include 10% of the data when training 
    test_support_ratio: how many training samples to be used when testing (meta support samples)
    inner_lr: inner optimization loop lr, smaller, e.g., 5e-5
    outer_lr: outer optimization loop lr, larger, e.g., 1e-5
    inner_steps_train: meta update steps when training, e.g., 5
    inner_steps_test: meta update steps when testing, smaller might be better, e.g., 2

    """
    reptile = ReptileWrapper(model, inner_lr, outer_lr, inner_steps_train, wandb_run)
    
    if use_simulation:
        if simulation_user_data is None:
            raise ValueError("simulation_user_data is required if use_simulation")
        train_user_data = simulation_user_data
        test_user_data = test_user_data
        
        best_f1 = 0
        try:
            for i in range(iteration_num+1):
                print(f"current iter num {i}")
                episode_data = random.sample(train_user_data, min(len(train_user_data), task_num_in_episode))
                episode = []
                for data in episode_data:
            
                
                    episode.append(BertTokenTask(data, tokenizer, support_ratio=train_support_ratio))
                episode_support_loss = reptile.meta_update(episode, support_batch_size=support_batch_size, query_batch_size=query_batch_size, max_support_subbatch_size=max_support_subbatch_size)
        
                if wandb_run is not None:
                    wandb_run.log({"episode_support_loss": np.mean(episode_support_loss)})
                if i % 100 == 0 and i != 0:
                    reptile_test = ReptileWrapper(model, inner_lr, outer_lr, inner_steps_test, wandb_run)
                    test_episode = []
                    for data in test_user_data:
                        test_episode.append(BertTokenTask(data, tokenizer, support_ratio=test_support_ratio)) 
                    test_episode_support_loss = reptile.meta_update(test_episode, support_batch_size=support_batch_size, query_batch_size=query_batch_size, max_support_subbatch_size=max_support_subbatch_size)
                    test_f1, test_acc, test_balanced_acc = evaluate_episode(test_episode, reptile_test)
                    if wandb_run is not None:
                        wandb_run.log({"test_f1": test_f1, "test_acc": test_acc, "test_balanced_acc": test_balanced_acc})
                    else:
                        print(f"test_f1: {test_f1}, test_acc: {test_acc}, test_balanced_acc: {test_balanced_acc}")
                    if save_model_path is not None:
                        if test_f1 > best_f1:
                            best_f1 = test_f1
                            reptile.save_model(save_model_path)
        finally:
            last_ckpt_path = Path(save_model_path).parent
            reptile.save_model(last_ckpt_path.joinpath("last_ckpt.pt"))

    else:
        print("using testing data only")
        for i in range(iteration_num):
            test_episode = []
            for data in test_user_data:
                test_episode.append(BertTokenTask(data, tokenizer, support_ratio=test_support_ratio))
            # meta_update first 
            episode_support_loss = reptile.meta_update(test_episode, support_batch_size=support_batch_size, query_batch_size=query_batch_size, max_support_subbatch_size=max_support_subbatch_size)
            if wandb_run is not None:
                wandb_run.log({"meta_update_loss": np.mean(episode_support_loss)})
            if i % 50 == 0:
                test_f1, test_acc, test_balanced_acc = evaluate_episode(test_episode, reptile)
                if wandb_run is not None:
                    wandb_run.log({"test_f1": test_f1, "test_acc": test_acc, "test_balanced_acc": test_balanced_acc})
                else:
                    print(f"test_f1: {test_f1}, test_acc: {test_acc}, test_balanced_acc: {test_balanced_acc}")
            

            
def evaluate_episode(episode, reptile, return_clf_report = False):
    average_f1 = []
    average_acc = []
    average_balanced_acc = []
    clf_report_tasks = []
    for task in episode:
        evaluation_set = task.get_evaluation_set(task.query_set) # for reptile, we use only support for training 
        res, labels = reptile.evaluate(evaluation_set, 100)
        task_f1 =  f1_score(labels, res,  average="macro")

        clf_report = classification_report(y_true = labels, y_pred=res, digits = 4)
        clf_report_tasks.append(clf_report)

        acc = accuracy_score(labels, res)
        average_f1.append(task_f1)
        average_acc.append(acc)
        average_balanced_acc.append(balanced_accuracy_score(labels, res ))
    average_f1_epoch = np.mean(average_f1)
    average_acc_epoch = np.mean(average_acc)
    average_balanced_acc_epoch = np.mean(average_balanced_acc)
    if not return_clf_report: 
        return average_f1_epoch, average_acc_epoch, average_balanced_acc_epoch
    else:
         return average_f1_epoch, average_acc_epoch, average_balanced_acc_epoch, clf_report_tasks


# def train_with_sampling(model, tokenizer, human_user_data,
#           inner_lr, outer_lr, inner_steps, 
#           max_support_subbatch_size = 50, 
#           wandb_run = None, use_simulation = False, simulation_user_data = None,
#           initial_sampling_batch_size = 10, select_sampling_batch_size = 5, total_sampling_size = 50, 
#           initial_train_iter_num = 10, iteration_num = 10, evaluate_batch_size = 200, load_model_path = None):
    
#     reptile = ReptileWrapper(model, inner_lr, outer_lr, inner_steps, wandb_run)
#     if load_model_path is not None:
#         reptile.load_model(load_model_path)
#         print(f"Loaded model from {load_model_path}")

#     if use_simulation:
#         if load_model_path is None:
#             raise ValueError("load_model_path is required if use_simulation")

#         # First trian the model with simulation data 
#         # Using the trained model for inference 
#         print("using simulation data")
#         test_episode = []
#         # load data as tasks
#         for data in human_user_data:
#             test_episode.append(BertTokenTask(data, tokenizer, support_ratio=initial_sampling_batch_size))
#         test_f1, test_acc, test_balanced_acc = evaluate_episode(test_episode, reptile)
#         if wandb_run is not None:
#             wandb_run.log({"initial_test_f1": test_f1, "initial_test_acc": test_acc, "initial_test_balanced_acc": test_balanced_acc})
#         else:
#             print(f"initial_test_f1: {test_f1}, initial_test_acc: {test_acc}, initial_test_balanced_acc: {test_balanced_acc}")
#         # train on the initial warm-up batch 
#         for i in range(initial_train_iter_num):
#             episode_support_loss = reptile.meta_update(test_episode, support_batch_size=initial_sampling_batch_size, query_batch_size=0, max_support_subbatch_size=max_support_subbatch_size)
#             if wandb_run is not None:
#                 wandb_run.log({"warmup_loss": np.mean(episode_support_loss)})
#         initial_f1, initial_acc, initial_balanced_acc = evaluate_episode(test_episode, reptile)
#         if wandb_run is not None:
#             wandb_run.log({"initial_f1": initial_f1, "initial_acc": initial_acc, "initial_balanced_acc": initial_balanced_acc})
#         else:
#             print(f"initial_f1: {initial_f1}, initial_acc: {initial_acc}, initial_balanced_acc: {initial_balanced_acc}")
#         # begin selecting samples 
#         for selecting_new_samples in range(initial_sampling_batch_size, total_sampling_size, select_sampling_batch_size): 
#             for idx, task in enumerate(test_episode):
#                 task.select_next_samples(reptile, num_samples=select_sampling_batch_size, evaluate_batch_size=evaluate_batch_size)
#                 test_episode[idx] = task
#             for iter in range(iteration_num):
#                 support_set_size = len(test_episode[0].support_set)
#                 episode_support_loss = reptile.meta_update(test_episode, support_batch_size=support_set_size, query_batch_size=0, max_support_subbatch_size=max_support_subbatch_size)
#             if wandb_run is not None:
#                     wandb_run.log({"episode_support_loss": np.mean(episode_support_loss)})
#             test_f1, test_acc, test_balanced_acc = evaluate_episode(test_episode, reptile)
#             if wandb_run is not None:
#                 wandb_run.log({"test_f1": test_f1, "test_acc": test_acc, "test_balanced_acc": test_balanced_acc})
#             else:
#                 print(f"test_f1: {test_f1}, test_acc: {test_acc}, test_balanced_acc: {test_balanced_acc}")
#     else: 
#         print("using human data only")
#         if load_model_path is not None:
#             raise ValueError("load_model_path is not allowed if use_simulation is False")
#         # First sampling a number of samples for the initial training 
#         initial_test_episode = []
#         for data in human_user_data:
#             initial_test_episode.append(BertTokenTask(data, tokenizer, support_ratio=initial_sampling_batch_size))
#         for i in range(initial_train_iter_num):
#             episode_support_loss = reptile.meta_update(initial_test_episode, support_batch_size=initial_sampling_batch_size, query_batch_size=0, max_support_subbatch_size=max_support_subbatch_size)
#             if wandb_run is not None:
#                 wandb_run.log({"warmup_loss": np.mean(episode_support_loss)})
#         # Then, select samples based on selection strategy 
#         for selecting_new_samples in range(initial_sampling_batch_size, total_sampling_size, select_sampling_batch_size): 
#             for idx, task in enumerate(initial_test_episode):
#                 task.select_next_samples(reptile, num_samples=select_sampling_batch_size, evaluate_batch_size=evaluate_batch_size)
#                 initial_test_episode[idx] = task
#             for iter in range(iteration_num):
#                 support_set_size = len(initial_test_episode[0].support_set)
#                 episode_support_loss = reptile.meta_update(initial_test_episode, support_batch_size=support_set_size, query_batch_size=0, max_support_subbatch_size=max_support_subbatch_size)
#             if wandb_run is not None:
#                     wandb_run.log({"episode_support_loss": np.mean(episode_support_loss)})
#             test_f1, test_acc, test_balanced_acc = evaluate_episode(initial_test_episode, reptile)
#             if wandb_run is not None:
#                 wandb_run.log({"test_f1": test_f1, "test_acc": test_acc, "test_balanced_acc": test_balanced_acc})
#             else:
#                 print(f"test_f1: {test_f1}, test_acc: {test_acc}, test_balanced_acc: {test_balanced_acc}")



def train_with_sampling(model, tokenizer, human_user_data,
          inner_lr, outer_lr, inner_steps, 
          max_support_subbatch_size = 50, 
          wandb_run = None, use_simulation = False, simulation_user_data = None,
          initial_sampling_batch_size = 10, select_sampling_batch_size = 5, total_sampling_size = 50, 
          initial_train_iter_num = 10, iteration_num = 10, evaluate_batch_size = 200, load_model_path = None):
    
    reptile = ReptileWrapper(model, inner_lr, outer_lr, inner_steps, wandb_run)

    if use_simulation:
        if load_model_path is None:
            raise ValueError("load_model_path is required if use_simulation")
        else:
            
            reptile.load_model(load_model_path)
            print(f"Loaded model from {load_model_path}")
        # First trian the model with simulation data 
        # Using the trained model for inference 
        print("using simulation data")
    else:
        raise ValueError("using human data only with sampling is not implemented yet")
        # print("using human data only")
    test_episode = []
    # load data as tasks
    for data in human_user_data:
        test_episode.append(BertTokenTask(data, tokenizer, support_ratio=initial_sampling_batch_size))
    if use_simulation:  
        test_f1, test_acc, test_balanced_acc = evaluate_episode(test_episode, reptile)
        if wandb_run is not None:
            wandb_run.log({"initial_test_f1": test_f1, "initial_test_acc": test_acc, "initial_test_balanced_acc": test_balanced_acc})
        else:
            print(f"initial_test_f1: {test_f1}, initial_test_acc: {test_acc}, initial_test_balanced_acc: {test_balanced_acc}")
    # train on the initial warm-up batch 
    for i in range(initial_train_iter_num):
        episode_support_loss = reptile.meta_update(test_episode, support_batch_size=initial_sampling_batch_size, query_batch_size=0, max_support_subbatch_size=max_support_subbatch_size)
        if wandb_run is not None:
            wandb_run.log({"warmup_loss": np.mean(episode_support_loss)})
    initial_f1, initial_acc, initial_balanced_acc = evaluate_episode(test_episode, reptile)
    if wandb_run is not None:
        wandb_run.log({"warmup_f1": initial_f1, "warmup_acc": initial_acc, "warmup_balanced_acc": initial_balanced_acc})
    else:
        print(f"warmup_f1: {initial_f1}, warmup_acc: {initial_acc}, warmup_balanced_acc: {initial_balanced_acc}")
    # begin selecting samples 
    # here, each user is updated differently 
    # first create a copy of the model 

    # update each user
    task_performance = {}
    # Create tmp folder if it doesn't exist
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    tmp_model_path = "temporary/reptile_tmp_ckpt.pt"
    reptile.save_model(tmp_model_path)
    for idx, task in enumerate(test_episode):
        if idx != 0:
            reptile.load_model(tmp_model_path) # restore the model back to the initial state 
        task_performance[idx] = 0
        
        for selecting_new_samples in range(initial_sampling_batch_size, total_sampling_size, select_sampling_batch_size): 
            task.select_next_samples(reptile, num_samples=select_sampling_batch_size, evaluate_batch_size=evaluate_batch_size)
            for iter in range(iteration_num):
                support_set_size = len(task.support_set)
                training_tasks = [task]
                task_loss = reptile.meta_update(training_tasks, support_batch_size=support_set_size, query_batch_size=0, max_support_subbatch_size=max_support_subbatch_size)
                if wandb_run is not None:
                    wandb_run.log({"{} task_loss".format(idx): np.mean(task_loss)})
            task_f1, task_acc, task_balanced_acc = evaluate_episode(training_tasks, reptile)
            if task_f1 > task_performance[idx]:
                task_performance[idx] = task_f1
            if wandb_run is not None:
                wandb_run.log({"{} task f1".format(idx): task_f1, "{} task acc".format(idx): task_acc, "{} task balanced_acc".format(idx): task_balanced_acc})
            else:
                print(f"task {idx} f1: {task_f1}, task {idx} acc: {task_acc}, task {idx} balanced_acc: {task_balanced_acc}")
    # Delete temporary checkpoint file
    if os.path.exists(tmp_model_path):
        os.remove(tmp_model_path)
    # Remove tmp directory if empty
    if os.path.exists("./tmp") and not os.listdir("./tmp"):
        os.rmdir("./tmp")
    all_performance = []
    for idx in range(len(test_episode)):
        all_performance.append(task_performance[idx])
    average_performance = np.mean(all_performance)
    if wandb_run is not None:
        wandb_run.log({"average_performance": average_performance})
        wandb_run.log({"task_performance": all_performance})
        wandb_run.log({"task_performance_dict": task_performance})
    else:
        print(f"average performance: {average_performance}")

# warmup: make it individual, test w/imbalance loss/sampling
# select a small validation set for each user  
def train_with_imbalance_and_val(model, tokenizer, human_user_data,
          inner_lr, outer_lr, inner_steps, 
          max_support_subbatch_size = 50, 
          wandb_run = None, 
          initial_sampling_batch_size = 20, select_sampling_batch_size = 5, total_sampling_size = 50, 
          initial_train_iter_num = 5,  evaluate_batch_size = 200, load_model_path = None):

    reptile = ReptileWrapper(model, inner_lr, outer_lr, inner_steps, wandb_run)
    reptile.load_model(load_model_path)
    print(f"Loaded model from {load_model_path}")

    

    # for data in human_user_data:
    #     user_task = BertTokenTask(data, tokenizer, support_ratio=0.9)
    #     initial_support_input, _ = user_task.get_data(20, 100)
    #     initial_inputs.append(initial_support_input)
 
    all_performance = []
    best_iterations = []
    final_iter_performance = []
    for idx, data in enumerate(human_user_data):
        print(idx)
        reptile.load_model(load_model_path)
        # if idx == 6:
        initial_inputs = []
        
        
        user_task = BertTokenTask(data, tokenizer, support_ratio=initial_sampling_batch_size, upsample=False)
        
        user_validation_task = BertTokenTask(data, tokenizer, support_ratio=10)
        eval_task = [user_validation_task]
        # train the model on the individual user data of the first batch 
        initial_support_input, _ = user_task.get_data(initial_sampling_batch_size, 100)
        initial_inputs.append(initial_support_input)
        # small_val_input, _ = user_task.get_data(20, 100)
        # res, labels = reptile.evaluate(small_val_input, 10)
        # initial_val_f1 = f1_score(res, labels, average="macro")
        # print(f"initial val f1: {initial_val_f1}")
        # task_f1, task_acc, task_balanced_acc = evaluate_episode(eval_task, reptile)
        # print(f"initial task f1: {task_f1}, initial task acc: {task_acc}, initial task balanced_acc: {task_balanced_acc}")
        best_f1 = 0
        best_acc = 0
        best_balanced_acc = 0
        best_iter = 0
        for i in range(initial_train_iter_num):
        
            reptile.meta_step(initial_inputs, max_support_subbatch_size=max_support_subbatch_size)
    
            task_f1, task_acc, task_balanced_acc = evaluate_episode(eval_task, reptile)
            if task_f1 > best_f1:
                best_f1 = task_f1
                best_acc = task_acc
                best_balanced_acc = task_balanced_acc
                best_iter = i
            # print(f"iteration {i}, task f1: {task_f1}, task acc: {task_acc}, task balanced_acc: {task_balanced_acc}")
                # res, labels = reptile.evaluate(small_val_input, 10)
                # val_f1 = f1_score(res, labels, average="macro")
                # print(f"iteration {i}, val f1: {val_f1}")
            # for i in range()
        
        print(f"task best f1: {best_f1}, task best acc: {best_acc}, task best balanced_acc: {best_balanced_acc}")
        print(f"task performance: task f1: {task_f1}, task acc: {task_acc}, task balanced_acc: {task_balanced_acc}")
        all_performance.append(best_f1)
        best_iterations.append(best_iter)
        final_iter_performance.append(task_f1)
    average_performance = np.mean(all_performance)
    average_final_iter_performance = np.mean(final_iter_performance)
    print(f"average performance: {average_performance}")
    print(f"best iterations: {best_iterations}")
    print(f"average final iter performance: {average_final_iter_performance}")
    return 


def finetune_and_evaluate(model,  task, inputs_from_task,
                        inner_lr, outer_lr, inner_steps,                                  
                        train_iter_num = 10, evaluate = True):
    reptile = ReptileWrapper(model, inner_lr, outer_lr, inner_steps, wandb_run=None)
    episode = [task] # episode contains a list of tasks
    if evaluate:
        f1_query, acc_query, balanced_acc_query = evaluate_episode(episode, reptile)
    for i in range(train_iter_num): # samples for meta_step are a list of inputs
        reptile.meta_step([inputs_from_task],max_support_subbatch_size=100)
  
    if evaluate:
        finetuned_f1, finetuned_acc, finetuned_balanced_acc, clf_report = evaluate_episode(episode, reptile, return_clf_report = True)
        # print(f"initial f1: {f1_query}, initial acc: {acc_query}, initial balanced acc: {balanced_acc_query}")
        # print(f"finetuned f1: {finetuned_f1}, finetuned acc: {finetuned_acc}, finetuned balanced acc: {finetuned_balanced_acc}")
        return f1_query, finetuned_f1, clf_report[0]
        # return f1_query, acc_query, balanced_acc_query

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--num_labels", type=int, default=2)
    # # load data, load it manually instead of in params
    # parser.add_argument("--use_simulation", type=bool, default=True)
    # parser.add_argument("--simulation_user_data", type=str, default=None)
    # parser.add_argument("--human_user_data", type=str, default=None)

    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--save_model_path", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--support_batch_size", type=int, default=50, help="the number of samples in each support set")   
    # training params 
    parser.add_argument(
        "--train_support_ratio",
        type=float,
        default=0.99,
        help="Fraction of training samples per task to include during training (e.g., 0.1 = 0.1 of data)"
    )

    parser.add_argument(
        "--test_support_ratio",
        type=float,
        default=50,
        help="Fraction of training samples per task to use as meta-support samples during testing"
    )

    parser.add_argument(
        "--inner_lr",
        type=float,
        default=1e-5,
        help="Learning rate for the inner optimization loop (smaller value)"
    )

    parser.add_argument(
        "--outer_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the outer optimization loop (larger value)"
    )

    parser.add_argument(
        "--inner_steps_train",
        type=int,
        default=5,
        help="Number of inner loop update steps during training"
    )

    parser.add_argument(
        "--inner_steps_test",
        type=int,
        default=5,
        help="Number of inner loop update steps during testing"
    )

    parser.add_argument(
        "--iteration_num",
        type=int,
        default=500,
        help="Number of training loops"
    )

    parser.add_argument(
        "--task_num_in_episode",
        type=int,
        default=20,
        help="how many tasks in one episode"
    )

    args = parser.parse_args()
    
    ############ load model #############
    model = BertTokenClf(args.model_path, args.num_labels).to("cuda")
    if "roberta" in args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, max_length=512, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, max_length=512)

    
    ############ load data ###############
    data_path = "path/to/data/"

    # # load human user data
    # human_user_data = []
    # with open(data_path + "sentence_level_real_user_labels.json", "r") as f:
    #     for line in tqdm(f.readlines()):
    #         human_user_data.append(json.loads(line))
    # print(f"human user data number: {len(human_user_data)}")
    
    # load simulation data 
    ######### random noise data ##########
    # # random_noise_data = []
    # # random_noise_user_data = []
    # # with open("path/to/generated_outputs/random_generation/random_user_labels_sent_level.json", "r") as f:
    # #     for line in tqdm(f.readlines()):
    # #         random_noise_data.append(json.loads(line))
    # # user_names = [list(u.keys())[0] for u in random_noise_data]
    # # for name, data in zip(user_names, random_noise_data):
    # #     random_noise_user_data.append(data[name]) # one element is a list of data
    # # print(f"random noise data: {len(random_noise_data)}")
    # # simulation_user_data = random_noise_user_data
    
    ########### qwen-70b simulation data #############

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
        assert "user" not in name
        simulation_user_data.append(data[name]) # one element is a list of data
    print(f"simulation user data length : {len(simulation_user_data)}")


    random.Random(0).shuffle(simulation_user_data)
    train_size = int(len(simulation_user_data)*0.9)
    train_user_data = simulation_user_data[:train_size]
    test_user_data = simulation_user_data[train_size:]
    print(f"train user data number: {len(train_user_data)}")
    print(f"test user data number: {len(test_user_data)}")

    

    # inner_lr = 5e-6
    # # outer_lr = 1e-4
    # outer_lr = 1e-5
    # inner_steps = 5
    # train_support_ratio = 0.99
    # test_support_ratio = 100
    # iteration_num = 500
    # task_num_in_episode = 20

    # # wandb_run = None
    wandb_run = wandb.init(project="mete_learn_vocab_predict", name=args.wandb_run_name)
    save_model_path = args.save_model_path
    # use_simulation = True


    train(model = model, tokenizer = tokenizer, test_user_data = test_user_data, simulation_user_data = train_user_data, 
          iteration_num = args.iteration_num, task_num_in_episode = args.task_num_in_episode, 
          inner_steps_train = args.inner_steps_train, inner_steps_test =  args.inner_steps_test, 
          inner_lr = args.inner_lr, outer_lr = args.outer_lr, 
          train_support_ratio = args.train_support_ratio, test_support_ratio = args.test_support_ratio,
          support_batch_size = 50, query_batch_size = 0, max_support_subbatch_size = 20, 
          wandb_run = wandb_run, use_simulation = True, save_model_path = save_model_path)

    # python train_reptile.py --model_path path/to/model_checkpoints/modern-bert-large --save_model_path path/to/trained_checkpoints/modern-bert-large/best_ckpt.pt --wandb_run_name modern-bert-large --inner_steps_train 5 --inner_steps_test 2 --inner_lr 1e-5 --outer_lr 1e-4 --iteration_num 600
   

    ####################### train with selection #########################################
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = BertTokenClf("bert-base-uncased", 2)

    # inner_lr = 1e-5
    # outer_lr = 1e-4
    # inner_steps = 5

   # wandb_run = wandb.init(project="reptile", name="reptile_simulation_seqsample_30_10_100_load")
    # # wandb_run = None
    # max_support_subbatch_size = 50
    
    # use_simulation = True
    # simulation_user_data = None

    # initial_sampling_batch_size = 30
    # select_sampling_batch_size = 10
    # total_sampling_size = 100

    # initial_train_iter_num = 10
    # iteration_num = 5 # iteration for each selection 
    # evaluate_batch_size = 200
    # load_model_path = "./ckpt/reptile_simulation_pretrain.pt"
    # train_with_sampling(model = model, tokenizer = tokenizer, human_user_data = human_user_data,
    #       inner_lr = inner_lr, outer_lr = outer_lr, inner_steps = inner_steps, 
    #       max_support_subbatch_size = max_support_subbatch_size, 
    #       wandb_run = wandb_run, use_simulation = use_simulation, simulation_user_data = simulation_user_data,
    #       initial_sampling_batch_size = initial_sampling_batch_size, select_sampling_batch_size = select_sampling_batch_size, total_sampling_size = total_sampling_size,
    #       initial_train_iter_num = initial_train_iter_num, iteration_num = iteration_num, evaluate_batch_size = evaluate_batch_size, load_model_path = load_model_path)

# [0.736, 0.623, 0.728, 0.694, 0.73, 0.671, 0.66, 0.68, 0.548, 0.453, 0.554, 0.738, 0.730, 0.647, 0.737, 0.717] bert-large, 0.665

############ test the impact of testing labels imbalance ################
# up to 0.741 
    # def evaluate_task_resampled(task, reptile, resampled = True):

    #     evaluation_set = task.get_evaluation_set(task.query_set) # for reptile, we use only support for training 
    #     res, labels = reptile.evaluate(evaluation_set, 100)
    #     if resampled:
    #         # Get counts of each label
    #         label_0_indices = np.where(labels == 0)[0]
    #         label_1_indices = np.where(labels == 1)[0]
            
    #         # Get minimum count between 0s and 1s
    #         min_count = min(len(label_0_indices), len(label_1_indices))
            
    #         # Randomly sample min_count indices from each label
    #         sampled_0_indices = np.random.choice(label_0_indices, min_count, replace=False)
    #         sampled_1_indices = np.random.choice(label_1_indices, min_count, replace=False)
            
    #         # Combine indices and sort them to maintain order
    #         balanced_indices = np.sort(np.concatenate([sampled_0_indices, sampled_1_indices]))
            
    #         # Create balanced subset
    #         labels = labels[balanced_indices]
    #         res = res[balanced_indices]
    #     else:
    #         res = res 
    #         labels = labels
    #     task_f1 =  f1_score(res, labels, average="macro")
    #     task_acc = accuracy_score(res, labels)
    #     task_balanced_acc = balanced_accuracy_score(res, labels)
    
    #     print(f"Resampled task performance: task f1: {task_f1}, task acc: {task_acc}, task balanced_acc: {task_balanced_acc}")
    #     print(classification_report(res, labels))
    #     return task_f1, task_acc, task_balanced_acc
    
    # reptile = ReptileWrapper(model, inner_lr, outer_lr, inner_steps, None)
    # reptile.load_model("./ckpt/reptile_simulation_pretrain.pt")
    # # # task1 = BertTokenTask(human_user_data[5], tokenizer, support_ratio=50)
    # # task = BertTokenTask(human_user_data[6], tokenizer, support_ratio=50)
    # # task3 = BertTokenTask(human_user_data[4], tokenizer, support_ratio=50)
    # average_f1 = []
    # for idx, data in enumerate(human_user_data):
    #     print(f"evaluating task {idx}")
    #     reptile.load_model("./ckpt/reptile_simulation_pretrain.pt")
    #     task = BertTokenTask(data, tokenizer, support_ratio=50)
    #     # evaluate_task_resampled(task, reptile, resampled = False)
    #     task_f1 = 0
    #     for iter in range(2):
    #         training_tasks = [task]
    #         task_loss = reptile.meta_update(training_tasks, support_batch_size=50, query_batch_size=0, max_support_subbatch_size=50)
    #         f1, acc, balanced_acc = evaluate_task_resampled(task, reptile, resampled = True)
    #         if f1 > task_f1:
    #             task_f1 = f1
    #         # task_f1 = f1
    #     average_f1.append(task_f1)
    # print(f"average f1: {np.mean(average_f1)}")
            
                
############ train with imbalance and validation ################
    # inner_lr = 1e-5
    # outer_lr = 1e-4
    # inner_steps = 5
  
  

    # # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", max_length=512)
    # # model = BertTokenClf("bert-base-uncased", 2)

    # tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", max_length=512)
    # model = BertTokenClf("bert-large-uncased", 2)
    # # # wandb_run = wandb.init(project="reptile_bert_large", name="reptile_human_seqsample_10_5_50")
    # wandb_run = None
    # # load_model_path = "./ckpt/reptile_simulation_pretrain.pt"
    # load_model_path = "./ckpt/reptile_simulation_pretrain_bert_large.pt"
    # train_with_imbalance_and_val(model, tokenizer, human_user_data,
    #       inner_lr, outer_lr, inner_steps, 
    #       wandb_run = wandb_run, 
    #       initial_sampling_batch_size = 30, 
    #       initial_train_iter_num = 10, load_model_path = load_model_path) 


    # max_support_subbatch_size = 50

    # initial_sampling_batch_size = 20
    # select_sampling_batch_size = 5
    # total_sampling_size = 50

    # initial_train_iter_num = 10
    # iteration_num = 2 # iteration for each selection 
    # evaluate_batch_size = 200
    # load_model_path = "./ckpt/reptile_simulation_pretrain_bert_large.pt"
    # train_with_sampling(model = model, tokenizer = tokenizer, human_user_data = human_user_data,
    #       inner_lr = inner_lr, outer_lr = outer_lr, inner_steps = inner_steps, 
    #       max_support_subbatch_size = max_support_subbatch_size, 
    #       wandb_run = wandb_run, use_simulation = use_simulation, simulation_user_data = simulation_user_data,
    #       initial_sampling_batch_size = initial_sampling_batch_size, select_sampling_batch_size = select_sampling_batch_size, total_sampling_size = total_sampling_size,
    #       initial_train_iter_num = initial_train_iter_num, iteration_num = iteration_num, evaluate_batch_size = evaluate_batch_size, load_model_path = load_model_path)

