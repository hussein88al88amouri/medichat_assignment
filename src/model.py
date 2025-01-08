import torch
from unsloth import FastLanguageModel

def load_model(model_name, max_seq_length, dtype, load_in_4bit, device_map):
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")

def configure_peft_model(model, target_modules, lora_alpha=16, lora_dropout=0, random_state=3407, use_rslora=False):
    try:
        peft_model = FastLanguageModel.get_peft_model(
            model=model,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=None,
        )
        return peft_model
    except Exception as e:
        raise RuntimeError(f"Failed to configure PEFT model: {e}")
