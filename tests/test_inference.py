from src.inference import prepare_inference_inputs, generate_responses
from src.model import load_model
import pytest
import torch

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_gpu_feature():
    # Your test code that needs a GPU
    assert torch.cuda.is_available()

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
@pytest.fixture
def model_and_tokenizer():
    """Fixture to load model and tokenizer for inference"""
    model_name = "unsloth/Meta-Llama-3.1-8B"
    model, tokenizer = load_model(model_name, 16, None, True, {'': 0})
    return model, tokenizer

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_inference(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    
    # Test input values
    instruction = "What is your name?"
    input_text = "Tell me about yourself."
    eos_token = "<EOS>"

    # Prepare inference inputs
    inputs = prepare_inference_inputs(tokenizer, "Instruction: {}\nInput: {}", instruction, input_text, eos_token)
    
    # Generate responses
    responses = generate_responses(model, inputs, tokenizer, max_new_tokens=32)
    
    # Assertions
    assert isinstance(responses, list), f"Expected list, but got {type(responses)}"
    assert len(responses) > 0, "Expected non-empty responses list"
    assert isinstance(responses[0], str), f"Expected string, but got {type(responses[0])}"
    assert len(responses[0]) > 0, "Expected non-empty string response"

    # Optionally, assert that the response matches some expected pattern or content
    assert "name" in responses[0].lower(), "Response does not contain expected content"
