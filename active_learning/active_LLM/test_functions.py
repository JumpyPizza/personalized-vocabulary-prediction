import sys 
sys.path.append("../llm_inference")
sys.path.append("../meta_learn")

from generate import OpenAIGenerationModel
from bert_model import BertTokenTask, BertTokenClf 
from train_reptile import finetune_and_evaluate 

import json 
from tqdm import tqdm
from active_llm import UserSimulationOracle
from transformers import AutoTokenizer 
from prompts import RPOFILE_GENERALE_PROMPTS, RECOMMEND_PROMPTS_INITIAL, RECOMMEND_PROMPTS_CONTEXTUAL
from active_freq import LLMActiveLearning


from openai import OpenAI
import torch

# bert_base_uncased_path = "path/to/model_checkpoints/bert-base-uncased"
# user_file_path = "path/to/data/sentence_level_real_user_labels.json"

word_list_path = "../ckpt/word_list.txt"
bert_base_uncased_path = "bert-base-uncased"
user_file_path = "../ckpt/sentence_level_real_user_labels.json"
LOAD_MODEL_PATH = "../ckpt/reptile_simulation_pretrain.pt"

with open(word_list_path, "r", encoding="utf-8") as f: 
    word_list = [line.strip().lower() for line in f.readlines()]
    
human_user_data = []
with open(user_file_path, "r") as f:
    
    for idx, line in enumerate(tqdm(f.readlines())):
        human_user_data.append(json.loads(line))
        if idx >6:
            break
    
print(f"human user data size: {len(human_user_data)}")

tokenizer = AutoTokenizer.from_pretrained(bert_base_uncased_path)
user_6_data = human_user_data[6]

# =========== Load User Data ==========
user_oracle = UserSimulationOracle(user_data=user_6_data, tokenizer=tokenizer, initial_sample_num=30)
words = user_oracle.query_initial_samples()
print(words)
# ========== Load VLLM ===============
# vllm_model = VLLMGenerationModel(model_name_or_path="path/to/model_checkpoints/qwen-4b", gpu_num=1, max_tokens=2000, quantization=False)


# Configure an OpenAI client through environment variables when using this optional path.
# client = OpenAI()

# llm_model = OpenAIGenerationModel(client, "gpt-4o")
# ========== Load BertTokenClfModel ==========
prediction_model = BertTokenClf(bert_base_uncased_path, 2).cuda()
prediction_model.load_state_dict(torch.load(LOAD_MODEL_PATH))
# reptile_model = ReptileWrapper(model, inner_lr, outer_lr, inner_steps, wandb_run)
# reptile.load_model(load_model_path)
print("model loading finished")
active_pipeline = LLMActiveLearning( prediction_model, tokenizer,
                                    user_oracle,
                                    vocab_list = word_list)

# # ======== generate profiles ======
# profiles = active_pipeline._generate_user_profiles(words)
# print(profiles[0])


# ======== generate init rexommendations 
# recommendations = active_pipeline._get_llm_candidate_recommendations(initial_samples_string, [], None)
# active_pipeline.run_active_learning_simulation()
# conf_err = active_pipeline._get_confident_errors(recommendations)
# print(conf_err)
# print(second_rec)
