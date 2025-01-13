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
        {
            "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
            "model_path": "shared/mistral/7B_Instruct_v0.3",
        },
        {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "model_path": "shared/qwen2.5/7B_Instruct",
        },
        {
            "model_name": "Qwen/Qwen2.5-3B-Instruct",
            "model_path": "shared/qwen2.5/3B_Instruct",
        },
        {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_path": "shared/qwen2.5/0.5B_Instruct",
        },
    ]

    for model in models_to_download:
        download_model(model["model_name"], model["model_path"])

    print("All specified models have been downloaded successfully.")