from datasets import load_dataset

def formatting_prompts_func(examples, template, eos_token):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    # Format the examples using the provided template
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = template.format(instruction, input_text, output) + eos_token
        texts.append(text)

    # Return a dictionary with the formatted text
    return {"text": texts}

def load_and_prepare_dataset(dataset_name, nsamples, formatting_func, template, eos_token):
    # Load the dataset and prepare it by applying the formatting function
    dataset = load_dataset(dataset_name, split="train").select(range(nsamples))
    
    # Map the formatting function over the dataset
    return dataset.map(lambda examples: formatting_func(examples, template, eos_token), batched=True)
