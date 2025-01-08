from src.model import load_model, configure_peft_model
import torch

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_gpu_feature():
    # Your test code that needs a GPU
    assert torch.cuda.is_available()

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_load_model():
    model_name = "unsloth/Meta-Llama-3.1-8B"
    model, tokenizer = load_model(model_name, 16, None, True, {'': 0})

    # Check that model and tokenizer are not None
    assert model is not None
    assert tokenizer is not None
    
    # Check that model is on the correct device (e.g., GPU or CPU)
    assert next(model.parameters()).device == torch.device('cuda:0'), "Model should be loaded on CUDA device"

    # Check that the tokenizer is an instance of the correct class
    assert hasattr(tokenizer, "encode"), "Tokenizer should have the 'encode' method"

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_configure_peft_model():
    model_name = "unsloth/Meta-Llama-3.1-8B"
    model, _ = load_model(model_name, 16, None, True, {'': 0})

    # Configure the PEFT model
    peft_model = configure_peft_model(model, target_modules=["q_proj", "down_proj"])

    # Check that PEFT model is not None
    assert peft_model is not None, "PEFT model should not be None"
    
    # Check that the PEFT model has a forward method
    assert hasattr(peft_model, "forward"), "PEFT model should have a 'forward' method"

    # Ensure that PEFT model can perform a forward pass (check if no error is raised)
    try:
        dummy_input = torch.randint(0, 1000, (1, 16))  # Dummy input tensor
        peft_model(dummy_input)
    except Exception as e:
        pytest.fail(f"PEFT model forward pass failed: {e}")