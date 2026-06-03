from unsloth import FastLanguageModel
import torch
from datasets import load_from_disk
from trl import GRPOTrainer, GRPOConfig
from vllm import SamplingParams
from reward_function import reward_func, think_format_reward, match_format_approximately



model_dir = "path/to/model_checkpoints/qwen-4b"
dataset = load_from_disk("path/to/data/word_knowledege_dataset")
dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "user",   "content": x["prompt"],}
    ],
    "ground_truth": x["ground_truth"],
})

# 3k + 512 prompt for reasoning should be enough; penalize too long as it contains repetition 
max_seq_length = 4000 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower
max_prompt_length = 512
max_completion_length = max_seq_length - max_prompt_length



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_dir,
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
)





vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 0.95,
    top_k = -1,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
    repetition_penalty = 1.2,
    temperature = 1,
)

# unsloth doc uses vllm_sampling_params, but could not find it in trl
# trl seems to use generation_kwargs? 
# if vllm_sampling_params is working, it should override the params passed directly
training_args = GRPOConfig(
    # generation args
    vllm_sampling_params = vllm_sampling_params,
    min_p = 0.1,
    top_p = 0.95,
    top_k = -1,
    repetition_penalty = 1.2,
    temperature = 1,

    # learning rate
    learning_rate = 5e-6,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",

    # batch size, the larger the more stable
    per_device_train_batch_size = 12,
    gradient_accumulation_steps = 4, 
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,

    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 601,
    save_steps = 200,
    output_dir = "path/to/model_checkpoints/qwen4b_ckpt",

    # log
    # log_completions = True,
    # num_completions_to_print = 1, 
    # wandb_log_unique_prompts = True,
    logging_strategy = "steps",
    logging_steps = 1,
    report_to = "wandb",
    run_name = "init30_singleword_batch12_16users",
    

)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs=[reward_func,think_format_reward, match_format_approximately],
    args = training_args,
    train_dataset = dataset,
)

trainer.train()