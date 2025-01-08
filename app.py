import streamlit as st
# from unsloth import FastLanguageModel
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
from llama_cpp import Llama

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
    # Replace with the actual path to your GGUF model
    # Replace with your repository and model file name
    repo_id = "helamouri/medichat_assignment"
    filename = "llama3_medichat.gguf"
    # Download the file to a local path
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    # model_path = "helamouri/medichat_assignment/blob/main/llama3_medichat.gguf"
    return Llama(model_path=model_path)

# Generate a response using Llama.cpp
def generate_response(model, prompt):
    response = model(
        prompt, 
        max_tokens=200,  # Maximum tokens for the response
        temperature=0.7,  # Adjust for creativity (lower = deterministic)
        top_p=0.9,  # Nucleus sampling
        stop=["\n"]  # Stop generating when newline is encountered
    )
    return response["choices"][0]["text"]

# Load the model and tokenizer (GGUF format)
# @st.cache_resource
# def load_model():
#     model_name = "helamouri/lora_model"  # Replace with your model's GGUF path
#     model = FastLanguageModel.from_pretrained(model_name, device='cpu')  # Load the model using unsloth
#     tokenizer = model.tokenizer  # Assuming the tokenizer is part of the GGUF model object
#     return tokenizer, model


# @st.cache_resource
# def load_model():
#     model_name = "helamouri/medichat_assignment"  # Replace with your model's path
#     # Load the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # Load the model (if it's a causal language model or suitable model type)
#     model = AutoModelForCausalLM.from_pretrained(model_name,
#                                                  device_map="cpu",
#                                                  revision="main",
#                                                  quantize=False,
#                                                  load_in_8bit=False,
#                                                  load_in_4bit=False,
#                                                  torch_dtype=torch.float32
#                                                  )
#     return tokenizer, model

# Suppress unwanted outputs from unsloth or any other libraries during model loading
#suppress_output()

# Load the GGUF model
model = load_model()
# Restore stdout and stderr

#restore_output()

# App layout
st.title("MediChat: Your AI Medical Consultation Assistant")
st.markdown("Ask me anything about your health!")
st.write("Enter your symptoms or medical questions below:")

# User input
user_input = st.text_input("Your Question:")
if st.button("Get Response"):
    if user_input:
        with st.spinner("Generating response..."):
            # Generate Response
            response = generate_response(model, user_input)
        # Display response
        st.text_area("Response:", value=response, height=200)
    else:
        st.warning("Please enter a question.")
