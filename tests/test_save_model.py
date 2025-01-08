import os
import pytest
from src.save_model import save_model_and_tokenizer
from src.model import load_model
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
    """Fixture to load the model and tokenizer for saving."""
    model_name = "unsloth/Meta-Llama-3.1-8B"
    model, tokenizer = load_model(model_name, 16, None, True, {'': 0})
    return model, tokenizer

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_save_model(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    save_directory = "./test_save_dir"
    
    # Save model and tokenizer
    save_model_and_tokenizer(model, tokenizer, save_directory)
    
    # Check if the directory exists
    assert os.path.exists(save_directory), f"Directory {save_directory} does not exist"
    
    # Check for key model files
    assert os.path.exists(os.path.join(save_directory, "config.json")), "config.json not found"
    assert os.path.exists(os.path.join(save_directory, "tokenizer_config.json")), "tokenizer_config.json not found"
    assert os.path.exists(os.path.join(save_directory, "pytorch_model.bin")), "pytorch_model.bin not found"
    
    # Check that files are not empty
    assert os.path.getsize(os.path.join(save_directory, "pytorch_model.bin")) > 0, "pytorch_model.bin is empty"
    assert os.path.getsize(os.path.join(save_directory, "config.json")) > 0, "config.json is empty"
    assert os.path.getsize(os.path.join(save_directory, "tokenizer_config.json")) > 0, "tokenizer_config.json is empty"
    
    # Cleanup after test
    for file in os.listdir(save_directory):
        file_path = os.path.join(save_directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(save_directory)
