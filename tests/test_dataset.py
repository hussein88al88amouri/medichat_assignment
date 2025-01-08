from src.dataset import formatting_prompts_func

def test_formatting_prompts_func():
    # Test case with basic input
    examples = {
        "instruction": ["Test instruction"],
        "input": ["Test input"],
        "output": ["Test output"],
    }
    template = "Instruction: {}\nInput: {}\nOutput: {}"
    eos_token = "<EOS>"
    
    result = formatting_prompts_func(examples, template, eos_token)
    
    # Check if result contains the 'text' key
    assert "text" in result
    
    # Check if result contains exactly one formatted entry
    assert len(result["text"]) == 1
    
    # Check if the formatted text is correct
    expected = "Instruction: Test instruction\nInput: Test input\nOutput: Test output<EOS>"
    assert result["text"][0] == expected

    # Test with empty inputs (edge case)
    examples_empty = {
        "instruction": [""],
        "input": [""],
        "output": [""],
    }
    result_empty = formatting_prompts_func(examples_empty, template, eos_token)
    assert result_empty["text"][0] == "Instruction: \nInput: \nOutput: <EOS>"

    # Test with multiple examples
    examples_multi = {
        "instruction": ["Test instruction 1", "Test instruction 2"],
        "input": ["Test input 1", "Test input 2"],
        "output": ["Test output 1", "Test output 2"],
    }
    result_multi = formatting_prompts_func(examples_multi, template, eos_token)
    assert len(result_multi["text"]) == 2
    assert result_multi["text"][0] == "Instruction: Test instruction 1\nInput: Test input 1\nOutput: Test output 1<EOS>"
    assert result_multi["text"][1] == "Instruction: Test instruction 2\nInput: Test input 2\nOutput: Test output 2<EOS>"
