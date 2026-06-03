import json
from sklearn.metrics import f1_score, classification_report
import numpy as np

############### qwq ################
output_file = "./output/qwq/{}.json"
all_results = []
for i in range(16):
    with open(output_file.format(i), 'r') as f:
        results = {}
        for line in f:
            data = json.loads(line)
            try:
                answer = data.split("</think>\n\n")[1]
            except IndexError:
                # print(data)
                continue
            try:
                answer = json.loads(answer)
            except:
                answer = answer.split("\n\n")[0]
                answer = json.loads(answer)
            for word, pred in answer.items():
                results[word] = pred

    with open(f"./output/qwq/{i}_label.json", 'r') as f:
        label = json.load(f)
    pred_labels = []
    true_labels = []
    for word in results.keys():
        pred_res = {"yes": 1, "no": 0}[results[word]]
        try:
            true_res = {5: 1, 4:1, 3:0, 2:0, 1:0}[label[word.lower().lstrip().rstrip()]]
        except:
            word = word.lower().lstrip().rstrip()
            word = word.split(" ")[1]
            true_res = {5: 1, 4:1, 3:0, 2:0, 1:0}[label[word]]
        pred_labels.append(pred_res)
        true_labels.append(true_res)
    # print("total pred number of 1", sum([l for l in pred_labels if l == 1]))
    # print("total true number of 1", sum([l for l in true_labels if l == 1]))
    print(i)
    print(f1_score(true_labels, pred_labels, average="macro"))
    all_results.append(f1_score(true_labels, pred_labels, average="macro"))
print(np.mean(all_results))
################ qwen ################
# output_file = "./output/qwen/{}.json"
# all_results = []
# for i in range(16):
#     with open(output_file.format(i), 'r') as f:
#         results = {}
#         for line in f:
#             data = json.loads(line)
#             data = json.loads(data)
#             for word, pred in data.items():
#                 results[word] = pred

#     with open(f"./output/qwen/{i}_label.json", 'r') as f:
#         label = json.load(f)
#     pred_labels = []
#     true_labels = []
#     for word in results.keys():
#         pred_res = {"yes": 1, "Yes": 1, "Know":1, "know":1, "no": 0, "not know": 0, "No": 0,  "Not know": 0, "might": 1, "Might": 1}[results[word]]
#         try:
#             true_res = {5: 1, 4:1, 3:0, 2:0, 1:0}[label[word.lower().lstrip().rstrip()]]
#         except:
#             raise ValueError(f"word {word} not found in label")
#             # word = word.lower().lstrip().rstrip()   
#             # word = word.split(" ")[1]
#             # true_res = {5: 1, 4:1, 3:0, 2:0, 1:0}[label[word]]
#         pred_labels.append(pred_res)
#         true_labels.append(true_res)
#     print(i)
#     print(f1_score(true_labels, pred_labels, average="macro"))
#     all_results.append(f1_score(true_labels, pred_labels, average="macro"))
# print(np.mean(all_results))