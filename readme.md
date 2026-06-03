# Personalized Vocabulary Knowledge Prediction

This repository contains research code used for the experiments in **Personalized Vocabulary Knowledge Prediction for Second-language Learners**.
Please note that the original datasets and licensed corpora are not redistributed in this repository.

## Code and Paper Alignment


| Paper component | Code location |
| --- | --- |
| L2 learner profile and vocabulary-knowledge simulation | `simulation_experiments/`, `llm_inference/`, `data_utils/data_pipeline.py` |
| Context extraction and sentence-level label construction | `data_utils/preprocess_sentence.py`, `data_utils/create_token_labels.py`, `training/build_dataset_sent_level_split.py` |
| Cross-learner transfer learning with masked language models | `training/train_bert_sent_level_split.py`, |
| Personalized adaptation for unseen learners | `training/finetune_bert.py`,  |
| Frequency-band and feature baselines | `baselines/feature_baseline/` |
| Graph-based LLGC baseline | `baselines/llgc/` |
| Deep IRT baseline | `baselines/irt.py` |
| Active-learning strategies | `active_learning/` |


## Data Used

These data are used in the experiments, but are not redistributed in this repo:

| Resource | Used for | Original source |
| --- | --- | --- |
| COCA, Corpus of Contemporary American English | Contextual sentences for target words and frequency-derived features | [English-Corpora.org COCA](https://www.english-corpora.org/coca/) |
| FCE learner corpus | L2 learner passages for simulation prompts | [CLC FCE dataset reference](http://ilexir.co.uk/applications/clc-fce-dataset/) and the original 2011 FCE learner-text paper |
| ICNALE | L2 learner passages for simulation prompts | [The ICNALE official site](https://language.sakura.ne.jp/icnale/) |
| TECCL | L2 learner passages for simulation prompts | [TECCL Corpus, BFSU Corpus Linguistics](https://corpus.bfsu.edu.cn/info/1070/1449.htm) |
| SVD12K | Main evaluation set | [COLING 2012 paper: Mining Words in the Minds of Second Language Learners](https://aclanthology.org/C12-1049/) |
| EVKD | Additional evaluation set | [LREC 2018 paper: Building an English Vocabulary Knowledge Dataset](https://aclanthology.org/L18-1076/) |

The paper binarizes SVD12K labels by mapping scores 1-3 to unknown and 4-5 to known. EVKD multiple-choice responses are converted to binary labels based on correctness.


## Extra experiments that fail
This repo also includes some experiments that we have tried but did not yield good performance, and are not reported in the paper. 
1. RL in active learning:  
RL appears a natural method for to train models that can select the best samples for a small budget. However, we tried several RL algorithms and none of them converge. 
2. Active LLM. 
As LLM understands semantics well, we might use LLM to suggest the next sample for training. However, this did not improve the performance. 
3. RLVR for reasoning model. 
In our experiments, we observe that reasoning model produces better results, so does RLVR improve the performance? Experiments show that RLVR does not converge. The signal is too sparse. 



