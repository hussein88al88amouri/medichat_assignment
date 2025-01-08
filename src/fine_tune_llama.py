from src import *

# Load configuration
max_seq_length = config.MAX_SEQ_LENGTH
device_map = config.DEVICE_MAP
eos_token = config.EOS_TOKEN

# Load and configure model
model_name = "unsloth/Meta-Llama-3.1-8B"
model, tokenizer = load_model(model_name, max_seq_length, config.DTYPE, config.LOAD_IN_4BIT, device_map)
eos_token = tokenizer.eos_token

model = configure_peft_model(model, target_modules=["q_proj", "down_proj"])

# Prepare dataset
nsamples = 1000
dataset = load_and_prepare_dataset(
    "lavita/ChatDoctor-HealthCareMagic-100k",
    nsamples,
    formatting_prompts_func,
    config.ALPACA_PROMPT_TEMPLATE,
    eos_token,
)

# Train model
trainer_stats = train_model(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    training_args=config.TRAIN_ARGS,
)

# Save the model
save_model_and_tokenizer(model, tokenizer, "./llama3_medichat")
