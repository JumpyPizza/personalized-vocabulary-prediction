from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json
from typing import List


class VLLMGenerationModel():
    def __init__(
        self,
        model_name_or_path: str,
        gpu_num: int,
        max_tokens: int = 1000,
        quantization=False,
        temperature=None,
        enable_lora=False,
        lora_path=None,
        **_,
    ):
        llm_kwargs = {
            "model": model_name_or_path,
            "tensor_parallel_size": gpu_num,
        }
        if quantization:
            print("using fp8 quantization")
            llm_kwargs["quantization"] = "fp8"
        if enable_lora:
            if not lora_path:
                raise ValueError("lora_path must be provided when enable_lora=True.")
            llm_kwargs["enable_lora"] = True
            from vllm.lora.request import LoRARequest
            self.lora_request = LoRARequest("adapter", 1, lora_path)
        else:
            self.lora_request = None

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        generation_config = json.load(open(os.path.join(model_name_or_path, "generation_config.json")))

        self.sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else generation_config.get("temperature", 0.6),
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
        
        generate_kwargs = {"sampling_params": self.sampling_params}
        if self.lora_request is not None:
            generate_kwargs["lora_request"] = self.lora_request
        outputs = self.llm.generate(inputs, **generate_kwargs)
        generated_texts = []
        for output in outputs:
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
        return generated_texts

