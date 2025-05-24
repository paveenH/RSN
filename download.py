#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on [Current Date]

Script to download specified language models and their tokenizers.

@author:
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def download_model(model_name, model_path):
    """
    Downloads the specified model and tokenizer from Hugging Face and saves them locally.

    Args:
        model_name (str): The Hugging Face model identifier.
        model_path (str): The local directory path to save the model and tokenizer.
    """
    cache_dir = os.path.join(os.getcwd(), model_path)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading model '{model_name}' to '{cache_dir}'...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,  # Specifies where to cache the downloaded files
            torch_dtype='float16',  # Adjust dtype as needed; 'float32' is default
            device_map="auto"  # Automatically maps the model to available devices
        )
    except Exception as e:
        print(f"Error downloading model '{model_name}': {e}")
        return

    print(f"Downloading tokenizer for '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=False  # Set to True if a fast tokenizer is available
        )
    except Exception as e:
        print(f"Error downloading tokenizer for '{model_name}': {e}")
        return

    # Explicitly save the model and tokenizer to the specified directory
    print(f"Saving model and tokenizer to '{cache_dir}'...")
    try:
        model.save_pretrained(cache_dir)
        tokenizer.save_pretrained(cache_dir)
    except Exception as e:
        print(f"Error saving model/tokenizer for '{model_name}': {e}")
        return

    print(f"Model and tokenizer for '{model_name}' downloaded and saved successfully.\n")


if __name__ == "__main__":
    models_to_download = [
        # {
        #     "model_name": "microsoft/Phi-3.5-mini-instruct",
        #     "model_path": "shared/phi/3.8B",
        # },
        # {
        #     "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        #     "model_path": "shared/mistral/7B",
        # },
        # {
        #     "model_name": "Qwen/Qwen2.5-7B-Instruct",
        #     "model_path": "shared/qwen2.5/7B",
        # },
        # {
        #     "model_name": "Qwen/Qwen2.5-3B-Instruct",
        #     "model_path": "shared/qwen2.5/3B",
        # },
        # {
        #     "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        #     "model_path": "shared/qwen2.5/0.5B",
        # },
        # {
        #     "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        #     "model_path": "shared/llama3/1B",
        # },
        # {
        #     "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        #     # "model_path": "shared/llama3/8B",
        # },
        # {
        #     "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        #     "model_path": "shared/llama3/3B",
        # },
        # {
        #     "model_name": "deepseek-ai/deepseek-llm-7b-base",
        #     "model_path": "shared/deepseek/7B",
        # }
        # {
        #     "model_name": "tiiuae/Falcon3-10B-Instruct",
        #     "model_path": "shared/falcon3/10B",
        # },
        # {
        #     "model_name": "tiiuae/Falcon3-7B-Instruct",
        #     "model_path": "shared/falcon3/7B",
        # },
        # {
        #     "model_name": "tiiuae/Falcon3-3B-Instruct",
        #     "model_path": "shared/falcon3/3B",
        # },
        # {
        #     "model_name": "tiiuae/Falcon3-1B-Instruct",
        #     "model_path": "shared/falcon3/1B",
        # },
        # {
        #     "model_name": "google/gemma-7b-it",
        #     "model_path": "shared/gemma/7B",
        # },
        {
            "model_name": "google/gemma-3b-4b-it",
            "model_path": "shared/gemma3/4B",
        },
        # {
        #     "model_name": "google/gemma-3b-1.1b-it",
        #     "model_path": "shared/gemma3/1B",
        # },
        # {
        #     "model_name": "google/gemma-3b-12b-it",
        #     "model_path": "shared/gemma3/12B",
        # },
        
    ]

    for model in models_to_download:
        download_model(model["model_name"], model["model_path"])

    print("All specified models have been downloaded successfully.")