---
title: MediChat
emoji: 🩺
colorFrom: blue
colorTo: yellow
sdk: streamlit  
sdk_version: "1.40.1"  # Replace with the actual version of your SDK
app_file: app.py  # Replace with the main app file name
pinned: false
---

[![CI/CD Workflow](https://github.com/hussein88al88amouri/medichat_assignment/actions/workflows/deploy.yml/badge.svg)](https://github.com/hussein88al88amouri/medichat_assignment/actions/workflows/deploy.yml)

# MediChat: AI-Powered Medical Consultation Assistant

MediChat is an intelligent chatbot designed to provide medical consultations using a fine-tuned Llama3.1:8B model. The project bridges advanced AI capabilities with practical healthcare assistance.

## Features
- Fine-tuned model for medical conversations
- Interactive and user-friendly interface
- Secure and containerized deployment

## How to Use
1. Access the chatbot interface.
2. Input your medical query.
3. Receive intelligent and context-aware responses.

## Technical Details
- Model: Llama3.1:8B
- Framework: Gradio

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/medichat.git
   cd medichat
2. Build and run the Docker container: (in bash copy the following code)
   docker build -t medichat-app .
   docker run -p 8501:8501 medichat-app
3. Access the app at http://localhost:8501.

Limitations
This tool is not a replacement for professional medical advice.
For critical issues, always consult a licensed medical professional.

