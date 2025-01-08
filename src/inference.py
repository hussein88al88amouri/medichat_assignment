from transformers import TextStreamer

def prepare_inference_inputs(tokenizer, template, instruction, input_text, eos_token, device="cuda"):
    """
    Prepares the inputs for inference by formatting the prompt and tokenizing it.

    Args:
    - tokenizer: The tokenizer used for tokenization.
    - template: The template string for the prompt format.
    - instruction: The instruction to be included in the prompt.
    - input_text: The input to be included in the prompt.
    - eos_token: The end of sequence token.
    - device: The device for the model ('cuda' or 'cpu').

    Returns:
    - Tokenized inputs ready for inference.
    """
    prompt = template.format(instruction, input_text, "") + eos_token
    return tokenizer([prompt], return_tensors="pt").to(device)

def generate_responses(model, inputs, tokenizer, max_new_tokens=64):
    """
    Generates responses from the model based on the provided inputs.

    Args:
    - model: The pre-trained model for generation.
    - inputs: The tokenized inputs to generate responses.
    - tokenizer: The tokenizer used to decode the output.
    - max_new_tokens: The maximum number of tokens to generate.

    Returns:
    - Decoded responses from the model.
    """
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def stream_responses(model, inputs, tokenizer, max_new_tokens=128):
    """
    Streams the model's response using a text streamer.

    Args:
    - model: The pre-trained model for generation.
    - inputs: The tokenized inputs to generate responses.
    - tokenizer: The tokenizer used to decode the output.
    - max_new_tokens: The maximum number of tokens to generate.

    Returns:
    - Streams the output directly.
    """
    text_streamer = TextStreamer(tokenizer)
    model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens)
