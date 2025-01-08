from src.training import train_model
from src.model import load_model
from src.dataset import formatting_prompts_func
from datasets import Dataset
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
def mock_dataset():
    """Fixture to provide a mock dataset for training"""
    data = {
        "instruction": ["Test instruction 1", "Test instruction 2"],
        "input": ["Test input 1", "Test input 2"],
        "output": ["Test output 1", "Test output 2"]
    }
    formatted_data = formatting_prompts_func(data, template="Instruction: {}\nInput: {}\nOutput: {}", eos_token="<EOS>")
    return Dataset.from_dict(formatted_data)

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_train_model(mock_dataset):
    """Test to ensure the training model function works with a mock dataset"""
    
    # Load model
    model_name = "unsloth/Meta-Llama-3.1-8B"
    model, tokenizer = load_model(model_name, 16, None, True, {'': 0})

    # Training arguments
    training_args = {
        "max_steps": 1,
        "output_dir": "outputs"
    }

    # Train the model
    train_stats = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=mock_dataset,
        dataset_text_field="text",
        max_seq_length=16,
        dataset_num_proc=1,
        packing=False,
        training_args=training_args
    )
    
    # Assert that training statistics are returned
    assert train_stats is not None
    
    # Optionally, check for specific fields in `train_stats` (e.g., loss, global_step)
    # Since trainer.train() returns an object that has 'global_step' and 'train_loss', we can assert them
    assert hasattr(train_stats, "global_step")
    assert hasattr(train_stats, "train_loss")
    
    # For further validation, assert that the model directory was created (outputs directory)
    assert "outputs" in train_stats.args.output_dir
