#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:49:19 2024

@author: paveenhuang
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_path  = "shared/llama3/1B"

cache_dir = os.path.join(os.getcwd(), model_path)
os.makedirs(cache_dir, exist_ok=True)

print(f"Downloading model '{model_name}' to '{cache_dir}'...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    low_cpu_mem_usage=True,
    torch_dtype='float16',  
)

print(f"Downloading tokenizer for '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    use_fast=False  
)

model.save_pretrained(cache_dir)
tokenizer.save_pretrained(cache_dir)

print("Download successfully.")