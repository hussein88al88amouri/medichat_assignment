import streamlit as st
# from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os
import sys

# # Suppress unwanted outputs (e.g., from unsloth or other libraries)
# def suppress_output():
#     sys.stdout = open(os.devnull, 'w')  # Redirect stdout to devnull
#     sys.stderr = open(os.devnull, 'w')  # Redirect stderr to devnull

# def restore_output():
#     sys.stdout = sys.__stdout__  # Restore stdout
#     sys.stderr = sys.__stderr__  # Restore stderr

# Load the model (GGUF format)
@st.cache_resource
def load_model():
    # Define the repository and model filenames for both the base model and LoRA adapter
    base_model_repo = "helamouri/Meta-Llama-3.1-8B-Q8_0.gguf"
    base_model_filename = "Meta-Llama-3.1-8B-Q8_0.gguf"
    adapter_repo = "helamouri/medichat_assignment"
    # adapter_filename = "llama3_medichat.gguf"  # assuming adapter is also in safetensors format
    adapter_repo = "helamouri/model_medichat_finetuned_v1"

    # Download the base model and adapter model to local paths
    base_model_path = hf_hub_download(repo_id=base_model_repo, filename=base_model_filename)
    adapter_model_path = hf_hub_download(repo_id=adapter_repo, filename=adapter_filename)

    # Log paths for debugging
    print(f"Base model path: {base_model_path}")
    print(f"Adapter model path: {adapter_model_path}")

    # Load the full model (base model) and the adapter (LoRA)
    try:
        model = Llama(model_path=base_model_path) #, adapter_path=adapter_model_path)
        print("Model loaded successfully.")
    except ValueError as e:
        print(f"Error loading model: {e}")
        raise

    return model

# Generate a response using Llama.cpp
def generate_response(model, prompt):
    print('prompt')
    print(prompt)
    response = model(
        prompt, 
        max_tokens=200,  # Maximum tokens for the response
        temperature=0.7,  # Adjust for creativity (lower = deterministic)
        top_p=0.9,  # Nucleus sampling
        stop=["\n"]  # Stop generating when newline is encountered
    )
    print('response["choices"]')
    print(response["choices"])
    return response["choices"][0]["text"]

# Load the model and tokenizer (GGUF format)
# @st.cache_resource
# def load_model():
#     model_name = "helamouri/model_medichat_finetuned_v1"  # Replace with your model's GGUF path
#     model = FastLanguageModel.from_pretrained(model_name, device='cpu')  # Load the model using unsloth
#     tokenizer = model.tokenizer  # Assuming the tokenizer is part of the GGUF model object
#     return tokenizer, model


# @st.cache_resource
# def load_model():
#     model_name = "helamouri/model_medichat_finetuned_v1"  # Replace with your model's path
#     # Load the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # Load the model (if it's a causal language model or suitable model type)
#     model = AutoModelForCausalLM.from_pretrained(model_name,
#                                                  device_map="cpu",
#                                                  revision="main",
#                                                  quantize=False,
#                                                  load_in_8bit=False,
#                                                  load_in_4bit=False,
#                                                  #torch_dtype=torch.float32
#                                                  )
#     return tokenizer, model

# Suppress unwanted outputs from unsloth or any other libraries during model loading
#suppress_output()

# Load the GGUF model
print('Loading the model')
model = load_model()
# Restore stdout and stderr

#restore_output()

# App layout
print('Setting App layout')
st.title("MediChat: Your AI Medical Consultation Assistant")
st.markdown("Ask me anything about your health!")
st.write("Enter your symptoms or medical questions below:")

# User input
print(f'Setting user interface')
user_input = st.text_input("Your Question:")
if st.button("Get Response"):
    if user_input:
        with st.spinner("Generating response..."):
            # Generate Response
            response = generate_response(model, user_input)
            print('Response')
            print(response)
        # Display response
        st.text_area("Response:", value=response, height=200)
    else:
        st.warning("Please enter a question.")
