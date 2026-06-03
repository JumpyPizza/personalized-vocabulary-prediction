from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json 
from tqdm import tqdm 

device = "cuda"
model_dir = "path/to/model_checkpoints/qwen_2_7b"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device)
# lora_path = "path/to/model_checkpoints/qwen7b_ckpt_sft"
# model = PeftModel.from_pretrained(model, lora_path + "/checkpoint-1250")


###### load samples 
user_file_path = "path/to/data/sentence_level_real_user_labels.json"
human_user_data = []
with open(user_file_path, "r") as f:
    
    for idx, line in enumerate(tqdm(f.readlines())):
        human_user_data.append(json.loads(line))
        if idx >6:
            break
    
user_6_data = human_user_data[6]

# =========== Load User Data ==========
user_oracle = UserSimulationOracle(user_data=user_6_data, tokenizer=tokenizer, initial_sample_num=30)
words = user_oracle.query_initial_samples()
print(words)

# texts = []
# for prompt in [prompt_1, prompt_2]:
#     messages = [
#         {"role": "user", "content": prompt}
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     texts.append(text)
# model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# print(json.loads(response[1]))