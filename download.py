#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:49:19 2024

@author: paveenhuang
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def download_model(model_name, model_path):
    cache_dir = os.path.join(os.getcwd(), model_path)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading model '{model_name}' to '{cache_dir}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,  # This specifies where to cache the downloaded files
        torch_dtype='float16',
    )

    print(f"Downloading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=False
    )

    # Explicitly save the model and tokenizer to the specified directory
    print(f"Saving model and tokenizer to '{cache_dir}'...")
    model.save_pretrained(cache_dir)
    tokenizer.save_pretrained(cache_dir)

    print("Model and tokenizer downloaded and saved successfully.\n")


if __name__ == "__main__":
    models_to_download = [
        # {
        #     "model_name": "lmsys/vicuna-7b-v1.5",
        #     "model_path": "shared/vicuna/7B",
        # },
        {
            "model_name": "meta-llama/Llama-3.2-1B-Instruct",
            "model_path": "shared/llama3/1B",
        },
        {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "model_path": "shared/llama3/8B",
        },
        {
            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
            "model_path": "shared/llama3/3B",
        },
    ]

    for model in models_to_download:
        download_model(model["model_name"], model["model_path"])

    print("All specified models have been downloaded successfully.")
