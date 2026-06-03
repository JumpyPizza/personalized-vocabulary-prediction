from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
# from unsloth.chat_templates import train_on_responses_only


model_dir = "path/to/model_checkpoints/qwen_2_70b"
dataset = load_from_disk("path/to/data/qwen_sim_user_dataset")


lora_rank = 32
max_seq_length = 1000 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_dir,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    # load_in_8bit = True,
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



def generate_conversation(examples):
    prompt  = examples["prompt"]
    response = examples["completion"]
    role_system = [{'content':'you are an experienced English teacher', 'role':'system'}]
    conversations = [
        role_system + p + r  for p, r in zip(prompt, response)
    ]
    return { "conversations": conversations }

def format_with_template(examples):
    formatted_text = [
        tokenizer.apply_chat_template(conv, tokenize=False)
        for conv in examples["conversations"]
    ]
    return {"text": formatted_text}

dataset_formatted = dataset.map(generate_conversation, batched=True)
dataset_formatted = dataset_formatted.map(format_with_template, batched=True)



trainer = SFTTrainer(
    model = model,
    train_dataset = dataset_formatted,
    args = SFTConfig(
        dataset_text_field = "text",
        eos_token = tokenizer.eos_token,
        completion_only_loss = True, 
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, 
        learning_rate = 2e-4,
        logging_steps = 5,
        save_steps = 300,
        save_total_limit = 2, 

        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        output_dir = "path/to/model_checkpoints/qwen70b_ckpt_sft_lora",
        report_to = "none", # Use this for WandB etc
    ),
)



trainer.train()