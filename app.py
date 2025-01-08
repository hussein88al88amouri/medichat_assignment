import streamlit as st
# from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys

# Suppress unwanted outputs (e.g., from unsloth or other libraries)
def suppress_output():
    sys.stdout = open(os.devnull, 'w')  # Redirect stdout to devnull
    sys.stderr = open(os.devnull, 'w')  # Redirect stderr to devnull

def restore_output():
    sys.stdout = sys.__stdout__  # Restore stdout
    sys.stderr = sys.__stderr__  # Restore stderr

# Load the model and tokenizer (GGUF format)
# @st.cache_resource
# def load_model():
#     model_name = "helamouri/lora_model"  # Replace with your model's GGUF path
#     model = FastLanguageModel.from_pretrained(model_name, device='cpu')  # Load the model using unsloth
#     tokenizer = model.tokenizer  # Assuming the tokenizer is part of the GGUF model object
#     return tokenizer, model


@st.cache_resource
def load_model():
    model_name = "helamouri/medichat"  # Replace with your model's path
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load the model (if it's a causal language model or suitable model type)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32)
    return tokenizer, model

# Suppress unwanted outputs from unsloth or any other libraries during model loading
suppress_output()

# Load model and tokenizer
tokenizer, model = load_model()

# Ensure the model is in inference mode
model = FastLanguageModel.for_inference(model)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Restore stdout and stderr
restore_output()

# App layout
st.title("MediChat: Your AI Medical Consultation Assistant")
st.markdown("Ask me anything about your health!")
st.write("Enter your symptoms or medical questions below:")

# User input
user_input = st.text_input("Your Question:")
if st.button("Get Response"):
    if user_input:
        with st.spinner("Generating response..."):
            # Tokenize input and move the tensors to the appropriate device
            inputs = tokenizer(user_input, return_tensors="pt").to(device)
            
            # Generate response
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=200,
                    num_return_sequences=1,
                    use_cache=True
                )
            
            # Decode the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display response
        st.text_area("Response:", value=response, height=200)
    else:
        st.warning("Please enter a question.")
