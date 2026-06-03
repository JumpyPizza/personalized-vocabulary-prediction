from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json
from typing import List


class VLLMGenerationModel():
    def __init__(self, model_name_or_path: str, gpu_num: int, max_tokens: int = 1000, quantization = False):
        if quantization:
            print("using fp8 quantization")
            self.llm = LLM(model=model_name_or_path, tensor_parallel_size=gpu_num, quantization="fp8")
            # raise NotImplementedError("Pre-Quantization with tensor parallelism is not supported in VLLM")
        else:
            self.llm = LLM(model=model_name_or_path, tensor_parallel_size=gpu_num)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        generation_config = json.load(open(os.path.join(model_name_or_path, "generation_config.json")))

        self.sampling_params = SamplingParams(
            temperature=generation_config.get("temperature", 0.6),
            top_p=generation_config.get("top_p", 0.95),
            top_k=generation_config.get("top_k", 20),
            repetition_penalty=generation_config.get("repetition_penalty", 1.0),
            max_tokens=max_tokens
            )
    def generate(self, prompt_list: List[List[dict]]):
        inputs = self.tokenizer.apply_chat_template(
            prompt_list,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        generated_texts = []
        for output in outputs:
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
        return generated_texts



