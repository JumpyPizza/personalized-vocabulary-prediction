import sys 
sys.path.append("../../vocab-prediction")
import torch 
from tqdm import tqdm 
import json 
from sklearn.metrics import classification_report, f1_score
import numpy as np

from llm_inference.llm_evaluate import get_samples, get_prediction
from llm_inference.generate import VLLMGenerationModel 

from json_repair import repair_json


"""
Evaluate LLM's vocab prediction performance.
Input: Prompt; Initial Samples; Words to test; Ground-truth labels 
Output: classification report

"""
# map multi-scale label to binary label
label_text = {
    1: "0", 
    2: "0",
    3: "0",
    4: "1",
    5: "1"
}

eval_prompt_binary = """
You are given a small labeled sample of vocabulary words that indicate a second-language English learner's knowledge of these words.
If the learner knows the word, the label is 1; If the learer does not know the word, the labels is 0.
Your goal is to analyze this small set, identify the internal pattern, and infer the user's vocabulary knowledge characteristics.
Then, predict a binary label for any given word: 1 means knows the word and 0 means does not know the word. 
You should directly return your inference in the json format, e.g.,{{"word_0": "1", "word_1": "1", "word_2": "1", ...}}.
The samples are: {samples};
Infer the user's knowledge about the following words: {word_to_pred}. 
"""

def parse_model_response(response):
    good_string = repair_json(response)
    return json.loads(good_string)
    
def evaluate_llm(model_path, prompt_template, user_data_list, quantization = False,
                word_bs = 10, batch_size = 10, lora_path = None, temprature = 1 ):
    # word_bs: how many words to predict in one generation 
    gpu_num = torch.cuda.device_count()
    if lora_path:
        print("loading lora")
        llm = VLLMGenerationModel(model_name_or_path = model_path, gpu_num = gpu_num, temperature = 1,
                                quantization = quantization, enable_lora = True, lora_path = lora_path )
 
    else:
        llm = VLLMGenerationModel(model_name_or_path = model_path, gpu_num = gpu_num, temperature = 1,
                               quantization = quantization, enable_lora = False )
    # begin inference
    f1_list = []
    for idx, user_data in enumerate(user_data_list):
        train_data, eval_data = get_samples(user_data)
        # propcess train data based on the labeling mechanism (binary vs multi-class)
        train_data = [(word, label_text[label]) for word, label in train_data]
        learner_answers = json.dumps({word: label for word, label in train_data})
        # ground truth data 
        ground_truth = {}
        for word, label in eval_data:
            ground_truth[word] = label_text[label] # convert ground truth to the model predicted scale
        # prepare the test data, get them into prompts 
        words_to_pred = [w for w, l in eval_data]
        prompt_list = []
        for i in range(0, len(words_to_pred), word_bs):
            words_batch = words_to_pred[i:i+word_bs]
            prompt = prompt_template.format(samples = learner_answers, word_to_pred = words_batch)
            prompt_list.append(prompt)

        results = get_prediction(llm, prompt_list, batch_size = batch_size) # batch_size = how many prompts in one pass
        predictions = {}
        for result_dict in results:
            # result_dict: words: labels 
            try:
                model_res = parse_model_response(result_dict)
            except:
                print(result_dict)
            for w, l in model_res.items():
                predictions[w] = l
        
        
        true_labels = []
        pred_labels = []
        for w, l in ground_truth.items():
            if w not in predictions.keys():
                continue
            true_labels.append(l)
            pred_labels.append(predictions[w])

        with open("qwen_70b_results.txt", "a", encoding="utf-8") as f:
            f.write(str(idx))
            f.write("\n")
            f.write(classification_report(true_labels, pred_labels, digits=4))
            f.write("\n")
        print(classification_report(true_labels, pred_labels, digits=4))
        f1 = f1_score(true_labels, pred_labels, average = "macro")
        f1_list.append(f1)
    return f1_list

if __name__ == "__main__" : 
    # gpu_num = torch.cuda.device_count()
    model_path = "path/to/model_checkpoints/qwen_2_70b"
    lora_path = "path/to/model_checkpoints/qwen70b_ckpt_sft_lora/checkpoint-1250"

    # ###### load samples 
    user_file_path = "path/to/data/sentence_level_real_user_labels.json"
    human_user_data = []
    with open(user_file_path, "r") as f:
        
        for idx, line in enumerate(tqdm(f.readlines())):
            human_user_data.append(json.loads(line))
    
        


    f1_list = evaluate_llm(model_path = model_path, prompt_template = eval_prompt_binary, quantization = False,
            user_data_list = human_user_data, lora_path = lora_path, temprature = 1, batch_size=20 )
    print(f1_list)
    print(np.mean(f1_list))
