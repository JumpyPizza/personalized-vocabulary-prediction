import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4, 6, 7"
import sys
sys.path.append("path/to/repository")

import torch
import json
from tqdm import tqdm
from meta_learn.task_loader import IndividualTask
from generate import VLLMGenerationModel
from prompts import noreason_vocab_prediction_prompt
label_map = {
    1: "not know",
    2: "not know",
    3: "not know",
    4: "know",
    5: "know",
}

def get_word_label(data):
    word = data["word"]
    seq_labels = data['labels']
    label = None 
    for seq_label in seq_labels:
        for l in seq_label:
            if l != 0: 
                label = l 
                break 
    if label is not None:
        label = int(label)
        return word, label 
    else:
        return None, None 

def get_samples(data):
    task = IndividualTask(data, support_ratio=0.99)
    support_samples = task.sample_support(50)
    task = IndividualTask(data, support_ratio=0)
    query_samples = task.query_set
    train_data = []
    eval_data = []
    for sample in support_samples:
        word, label = get_word_label(sample)
        if word is not None:
            text_label = label_map[label]
            train_data.append((word, text_label))
            
    for sample in query_samples:
        word, label = get_word_label(sample)
        if word is not None:
            eval_data.append((word, label))
    return train_data, eval_data



def get_prediction(model, data, vocab_prediction_prompt, prompt_batch_size = 10, word_batch_size = 10):
    train_data, eval_data = get_samples(data)
    learner_answers = json.dumps({word: label for word, label in train_data})

    batch_input = []
    output_list = []
    ground_truth = {}
    for word, label in eval_data:
        ground_truth[word] = label
    for i in range(0, len(eval_data), word_batch_size): # how many words in each prompt 
        batch_data = eval_data[i:i+word_batch_size]
        new_words = [w for w, l in batch_data]

        prompt = vocab_prediction_prompt.format(learner_answers=json.dumps(learner_answers), new_words=json.dumps(new_words))

        # generate: list[list[dict]]
        input_text = [{"role": "user", "content": prompt}]
        batch_input.append(input_text)
    for i in range(0, len(batch_input), prompt_batch_size):
        print(f"current sample: {i}, progress: {i/len(batch_input)}")
        generated_texts = model.generate(batch_input[i:i+prompt_batch_size]) 
        for output in generated_texts:
            output_list.append(output)
    return output_list, ground_truth

if __name__ == "__main__":
    print("available gpus: ", torch.cuda.device_count())
    human_user_data = []
    with open("path/to/generated_outputs/coca_simulation/sentence_level_real_user_labels.json", "r") as f:
        for line in tqdm(f.readlines()):
            human_user_data.append(json.loads(line))
    print(f"human user data: {len(human_user_data)}")
    model = VLLMGenerationModel(model_name_or_path="path/to/local_models/qwen2.5-72b-instruct", gpu_num=torch.cuda.device_count(), max_tokens=2000, quantization=True)
    for idx, data in enumerate(human_user_data):
        
        output_list, ground_truth = get_prediction(model, data, noreason_vocab_prediction_prompt)
        with open(f"./output/qwen/{idx}.json", "w") as f:
            for o in output_list:
                f.write(json.dumps(o))
                f.write("\n")
        with open(f"./output/qwen/{idx}_label.json", "w") as f:
            f.write(json.dumps(ground_truth))