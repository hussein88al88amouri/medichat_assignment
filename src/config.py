import torch


def get_device_map():
    if torch.cuda.is_available():
        return {'' : torch.cuda.current_device()}
    else:
        return {}  # Or some default, fallback configuration

# General configuration
MAX_SEQ_LENGTH = 2**4
DTYPE = None
LOAD_IN_4BIT = True
DEVICE_MAP = {'': get_device_map()}
EOS_TOKEN = None  # Set dynamically based on tokenizer

# Alpaca prompt template
ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

###Instruction:
{}

###Input:
{}

###Response:
{}"""

# Training arguments
TRAIN_ARGS = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "max_steps": 60,
    "learning_rate": 2e-4,
    "fp16": not torch.cuda.is_bf16_supported(),
    "bf16": torch.cuda.is_bf16_supported(),
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "output_dir": "outputs",
}
