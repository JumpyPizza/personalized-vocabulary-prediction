import sys 
sys.path.append("../llm_inference")
sys.path.append("../meta_learn")

from bert_model import BertTokenTask, BertTokenClf 
from train_reptile import finetune_and_evaluate 
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch 
import numpy as np
import json
import re 

label_text = {
    1: "0", 
    2: "0",
    3: "0",
    4: "1",
    5: "1"
}

llm_label_text = {
    0: "does not know the word",
    1: "know the word"
}

user_binary_label_text = {
    0: "never seen the word before",
    1: "absolutely know the word's meaning"
}

class UserTruthOracle(ABC):
    """Abstract base class for getting ground truth from users"""
    
    @abstractmethod
    def query_initial_samples(self, num: int):
        """
        query the users for the initial samples; 
        """
        pass 
    @abstractmethod
    def get_ground_truth(self, samples: List[Dict[str, Any]]):
        """
        Get ground truth labels for the given samples from the user.
        
        Args:
            samples: List of word samples to query the user about
            
        Returns:
            List of ground truth labels corresponding to the samples
        """
        pass

class UserSimulationOracle(UserTruthOracle):
    """
    Query from the existing labels 
    """
    def __init__(self, user_data, tokenizer, initial_sample_num):
        self.initial_sample_num = initial_sample_num
        self.evaluation_loader = BertTokenTask(user_data, tokenizer, support_ratio=initial_sample_num)
        self.data_loader = BertTokenTask(user_data, tokenizer, support_ratio=0.99)
    
    def query_initial_samples(self, sample_num = None): 
        if sample_num is None:
            sample_num = self.initial_sample_num
        else:
            sample_num = sample_num
        # support_words : (words, labels)
    
       
        support_input, _, support_words = self.data_loader.get_data(support_batch_size = sample_num, query_batch_size = 0, ensure_non_zero = False)
               
  
        return support_words

    def get_ground_truth(self, samples):
        raise NotImplementedError
        
class LLMActiveLearning:
    """
    Input: initial samples, user labels; 
    1. Given the input, LLM generates a diverse set of user profiles;
    2. Given the input + user profile, LLM recommends several candidates that it thinks the user definitely knows or does not know; 
    3. Use the prediction model to evaluate those samples, get the most confident errors;
    4. Query the user with samples of the most confident errors; 
    5. Finetune the prediction model; input the queries to the LLM to recommend new samples; repeat;
    """
    def __init__(self,
                 llm_model,
                 prediction_model: BertTokenClf,
                 tokenizer, 
                 user_oracle, 
                 profile_generation_prompts, 
                 candidate_recommend_prompts_initial,
                 candidate_recommend_prompts_contextual,
                 vocab_list: List[str],
                 num_initial_samples: int = 20, 
                 n_query_numbers: int = 50, 
                 topk_confident_errors: int = 5,
                 ):
        self.llm_model = llm_model
        self.prediction_model = prediction_model
        self.tokenizer = tokenizer
        self.user_oracle = user_oracle
        self.profile_generation_prompts = profile_generation_prompts
        self.candidate_recommend_prompts_initial_only = candidate_recommend_prompts_initial
        self.candidate_recommend_prompts_contextual = candidate_recommend_prompts_contextual
        self.num_initial_samples = num_initial_samples
        self.n_query_samples = n_query_numbers
        self.topk_confident_errors = topk_confident_errors

        self.vocab_list = vocab_list
        
    
    def run_active_learning_simulation(self, use_profile = False ): 
        initial_samples = self.user_oracle.query_initial_samples() # [{"word": "apple",  "label: int}]
        # =========1. finetune the model with the initial samples ==============
        self._initial_finetune_eval(initial_samples)
        initial_sample_words = initial_samples[0]
        initial_sample_labels = initial_samples[1]
        initial_samples_info = {}
        for word, label in zip(initial_sample_words, initial_sample_labels):
            initial_samples_info[word] = label_text[label]
        initial_samples_string = json.dumps(initial_samples_info).lstrip("{").rstrip("}")

        if use_profile:
            self.user_profiles = self._generate_user_profiles(initial_samples_string)
        else:
            self.user_profiles = None 
        queried_samples = []
        # one element of queried_samples:
        # {'word': word,
        # 'LLM_label': llm_label,
        # 'confidence': confidence,
        # 'predicted_label': pred,
        # "ground_truth": true}
        # while len(queried_samples) < self.n_query_samples:
        for i in range(2):
            print(f"================= Round {i} =============")
            # get candidates from LLM
            candidate_samples = self._get_llm_candidate_recommendations(
                initial_samples_string, queried_samples, self.user_profiles, 
            )
            if not candidate_samples:
                raise RuntimeError("No candidate samples generated")
            # based on the candidates, get the topK confident errors, as the queries of this iteration
            try:
                confident_errors_samples = self._get_confident_errors(candidate_samples)
            except RuntimeError:
                raise RuntimeError(f"No matched words of LLM rec in the vocab list")


            # ground_truth_labels = self.user_oracle.get_ground_truth(confident_errors_samples)
            print(f"================= queried samples =============")
            print(confident_errors_samples)
            queried_samples.extend(confident_errors_samples)

            # update the prediction model using the queries of this iteration
            # TODO: compare update prediction model all at once vs sequentially 
            self._finetune_and_evaluation(initial_samples, queried_samples)
            
    def _generate_user_profiles(self, initial_samples: str) -> List[str]:

       
        
        prompt_list = []
        for prompt_template in self.profile_generation_prompts:
            formatted_prompt = prompt_template.format(samples = initial_samples)
            message = [{"role": "user", "content": formatted_prompt}]
            prompt_list.append(message)
        
        generated_profiles = self.llm_model.generate(prompt_list)
        return generated_profiles
    
    def _get_llm_candidate_recommendations(self, 
                                            initial_samples: str, 
                                            queried_samples: List[Dict[str, Any]], 
                                            user_profiles: None):
        #TODO: Run tests; initial round and following rounds;
        
        llm_recommendation_template = []
        if len(queried_samples) == 0:
            # Use initial query template 
            template = self.candidate_recommend_prompts_initial_only
            if user_profiles is None:
                llm_recommend_template = template.format(
                    initial_samples = initial_samples
                )
                llm_recommendation_template.append(llm_recommend_template)
        else: 
            # use following query template 
            template = self.candidate_recommend_prompts_contextual
            query_sample_text = ""
            for q_sample in queried_samples:
                word = q_sample['word']
                llm_label = q_sample['LLM_label']
                ground_label = q_sample['ground_truth']
                # TODO: check if the output text is correct
                text = f"word: {word}, previously inferred label by you: {llm_label}, label given by the learner: {ground_label}; "
                query_sample_text += text 
            print(query_sample_text)
            if user_profiles is None:
                llm_recommend_single_template = template.format(
                    initial_samples = initial_samples,
                    queried_samples = query_sample_text,
                )
             
                llm_recommendation_template.append(llm_recommend_single_template)

        chat_messages = []
        for prompt_template in llm_recommendation_template:
            chat_messages.append([{"role": "user", "content": prompt_template}])
        recommendations = self.llm_model.generate(chat_messages)
        # though we use batch chat_message here, since we only have one recommend template
        # we only get one element in the list 
        recommendations = recommendations[0]
        recommended_samples = self._parse_recommendations(recommendations)
        if len(recommended_samples) > 0 :
            return recommended_samples
        else:
            # raise error for now
            raise RuntimeError("no recommended samples")

    def _get_confident_errors(self, recommended_samples): 
        """
        based on the words and their labels given by the LLM
        get the prediction model's most confident errors as the query samples 
        Input: {"word": llm_label, ...}
        """
        inputs, llm_label_list, word_list = self._prepare_llm_sample_inputs(recommended_samples)
    
        if inputs is None: 
            raise RuntimeError("No recommended words found in the existing vocab")
        pred_labels, true_labels, pred_logits = self.prediction_model.evaluate_prediction(
            inputs, return_logits=True
        )
     
        confident_errors = []
        
        for i, (pred, true, logits, word, llm_label) in enumerate(zip(pred_labels, true_labels, pred_logits, word_list, llm_label_list)):
            # Calculate confidence as max probability
            probabilities = torch.softmax(torch.tensor(logits), dim=0)
            confidence = torch.max(probabilities).item()
            pred = int(pred)
            true = int(true) 
        
            # Get predictions that are different with the LLM, as pseudo errors 
            is_error = (pred != llm_label)
            
            if is_error:
                confident_errors.append({
                    'word': word,
                    'LLM_label': llm_label,
                    'confidence': confidence,
                    'predicted_label': pred,
                    "ground_truth": true
                })

        confident_errors.sort(key = lambda x:x['confidence'], reverse = True)
        # top_confident_errors = confident_errors[:self.topk_confident_errors]
        top_confident_errors = confident_errors
        return top_confident_errors
    
    def _prepare_llm_sample_inputs(self, samples):
        # the samples are from the llm recommendations 
        data_loader = self.user_oracle.data_loader
        samples_from_user = []
        llm_labels = []
        word_list = []
        # llm_label is a binary label 
        for word, llm_label in samples.items():
            sample = data_loader.get_sample_by_word(word)
            if sample is not None: 
                samples_from_user.append(sample)
                llm_labels.append(llm_label)
                word_list.append(word)
        assert len(word_list) == len(llm_labels) == len(samples_from_user)
        if len(samples_from_user) > 0 :
            sample_inputs = data_loader.get_data_by_samples(samples_from_user)
            return sample_inputs, llm_labels, word_list 
        else:
            return None 
         

    def _parse_recommendations(self, recommendations):
        # TODO: get samples and their labels from the LLM response 
        # TODO: what if samples not in the vocab 
        match = re.search(r"```json\s*([\s\S]*?)\s*```", recommendations, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON code block found in {}.".format(recommendations))

        json_like = match.group(1).strip()

        # If it's not a complete JSON object, wrap with braces
        if not json_like.startswith('{'):
            json_like = "{" + json_like + "}"

        llm_response = json.loads(json_like)
        
        valid_response = {}
        for word, label in llm_response.items():
            if word in self.vocab_list:
                valid_response[word] = label
        print("llm response")
        print(llm_response)
        
        return valid_response  
    
    def _initial_finetune_eval(self, initial_samples):
        
        eval_data_loader = self.user_oracle.evaluation_loader
        all_words = []
        for word in initial_samples[0]:
            all_words.append(word)
        finetune_samples = [self.user_oracle.data_loader.get_sample_by_word(w) for w in all_words]
        inputs = self.user_oracle.data_loader.get_data_by_samples(finetune_samples)
        initial_f1, finetine_f1 = finetune_and_evaluate(self.prediction_model,  eval_data_loader, inputs, 
                             inner_lr = 1e-5, outer_lr= 1e-4, inner_steps=5,                            
                             train_iter_num = 5)
        print(f"initial f1 is {initial_f1}, finetuned with initial samples f1 is {finetine_f1}")
        self.saved_state = self.prediction_model.state_dict()
        
    def _finetune_and_evaluation(self, initial_samples, queried_samples):
        # self.prediction_model.load_state_dict(self.saved_state)
        eval_data_loader = self.user_oracle.evaluation_loader
        # initial samples: a list of [word, labels]
        all_words = []
        # for word in initial_samples[0]:
        #     all_words.append(word)
        for sample in queried_samples:
            all_words.append(sample['word'])
        finetune_samples = [self.user_oracle.data_loader.get_sample_by_word(w) for w in all_words]
        inputs = self.user_oracle.data_loader.get_data_by_samples(finetune_samples)

        before_f1, finetune_f1 = finetune_and_evaluate(self.prediction_model,  eval_data_loader, inputs, 
                             inner_lr = 1e-5, outer_lr= 1e-4, inner_steps=5,                            
                             train_iter_num = 5)
        print(f"after this finetune round, the f1 is {finetune_f1}")
