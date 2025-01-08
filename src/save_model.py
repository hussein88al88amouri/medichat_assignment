import os

def save_model_and_tokenizer(model, tokenizer, save_directory):
    """
    Save model and tokenizer to the specified directory.

    Args:
    - model: The model to save.
    - tokenizer: The tokenizer to save.
    - save_directory: Directory where the model and tokenizer should be saved.
    """
    try:
        # Ensure the save directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(save_directory, safe_serialization=True)
        tokenizer.save_pretrained(save_directory)

        print(f"Model and tokenizer saved locally at {save_directory}")
    except Exception as e:
        print(f"Error saving model and tokenizer: {str(e)}")
        raise
