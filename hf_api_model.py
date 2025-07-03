# hf_api_model.py
from huggingface_hub import InferenceClient
import os

class HfApiModel:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", token=None):
        token = token or os.getenv("HF_API_TOKEN")
        self.client = InferenceClient(model=model_name, token=token)

    def chat(self, messages):
        # Concatenate previous messages into a prompt
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        # Add assistant prompt
        prompt += "Assistant:"

        # Send to Hugging Face Inference API
        response = self.client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            stop_sequences=["User:", "Assistant:"],
        )

        # Response is a single string
        return {"message": {"content": response.strip()}}
