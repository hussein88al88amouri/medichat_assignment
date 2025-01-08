import pytest
import torch
from src.config import (MAX_SEQ_LENGTH, DTYPE, LOAD_IN_4BIT, DEVICE_MAP, EOS_TOKEN, 
                        ALPACA_PROMPT_TEMPLATE, TRAIN_ARGS)

# Test that required configuration keys are present
def test_required_config_keys():
    assert MAX_SEQ_LENGTH is not None, "MAX_SEQ_LENGTH is not set."
    assert TRAIN_ARGS is not None, "TRAIN_ARGS is not set."
    assert ALPACA_PROMPT_TEMPLATE is not None, "ALPACA_PROMPT_TEMPLATE is not set."
    assert DEVICE_MAP is not None, "DEVICE_MAP is not set."

# Test that MAX_SEQ_LENGTH is a power of two
def test_max_seq_length():
    assert isinstance(MAX_SEQ_LENGTH, int), "MAX_SEQ_LENGTH should be an integer."
    assert MAX_SEQ_LENGTH > 0, "MAX_SEQ_LENGTH should be greater than 0."
    assert (MAX_SEQ_LENGTH & (MAX_SEQ_LENGTH - 1)) == 0, "MAX_SEQ_LENGTH should be a power of two."

# Test that TRAIN_ARGS dictionary contains required fields and types
def test_train_args():
    required_keys = [
        "per_device_train_batch_size", 
        "gradient_accumulation_steps", 
        "warmup_steps", 
        "max_steps", 
        "learning_rate", 
        "fp16", 
        "bf16", 
        "logging_steps", 
        "optim", 
        "weight_decay", 
        "lr_scheduler_type", 
        "seed", 
        "output_dir"
    ]
    
    for key in required_keys:
        assert key in TRAIN_ARGS, f"Missing {key} in TRAIN_ARGS."
    
    # Check types of specific fields
    assert isinstance(TRAIN_ARGS["per_device_train_batch_size"], int), "per_device_train_batch_size should be an integer."
    assert isinstance(TRAIN_ARGS["learning_rate"], float), "learning_rate should be a float."
    assert isinstance(TRAIN_ARGS["output_dir"], str), "output_dir should be a string."

# Test that the DEVICE_MAP references a valid CUDA device
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_device_map():
    device = DEVICE_MAP.get('', None)
    assert device is not None, "DEVICE_MAP should reference a CUDA device."
    assert isinstance(device, int), "DEVICE_MAP should be an integer (CUDA device ID)."
    assert torch.cuda.is_available(), "CUDA is not available, but DEVICE_MAP points to a CUDA device."

# Test that the EOS_TOKEN is set dynamically based on the tokenizer
def test_eos_token():
    assert EOS_TOKEN is not None, "EOS_TOKEN should be dynamically set based on tokenizer."

# Test the ALPACA_PROMPT_TEMPLATE for expected formatting
def test_alpaca_prompt_template():
    test_instruction = "Test Instruction"
    test_input = "Test Input"
    test_output = "Test Output"
    
    formatted_prompt = ALPACA_PROMPT_TEMPLATE.format(test_instruction, test_input, test_output)
    
    # Ensure that the prompt template contains the required placeholders
    assert "{}" in formatted_prompt, "ALPACA_PROMPT_TEMPLATE should contain placeholders."
    assert "###Instruction:" in formatted_prompt, "ALPACA_PROMPT_TEMPLATE should contain '###Instruction'."
    assert "###Input:" in formatted_prompt, "ALPACA_PROMPT_TEMPLATE should contain '###Input'."
    assert "###Response:" in formatted_prompt, "ALPACA_PROMPT_TEMPLATE should contain '###Response'."

# Test that the LOAD_IN_4BIT setting is a boolean
def test_load_in_4bit():
    assert isinstance(LOAD_IN_4BIT, bool), "LOAD_IN_4BIT should be a boolean."

# Test for the DTYPE (should be None or a valid data type)
def test_dtype():
    assert DTYPE is None or isinstance(DTYPE, type), "DTYPE should be None or a valid data type."

