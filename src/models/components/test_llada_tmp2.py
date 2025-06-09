#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 22:00:48 2025

@author: paveenhuang
"""

from transformers import AutoModel, AutoTokenizer
import torch
from generate import generate as gen_diffusion  

model_name = "GSAI-ML/LLaDA-1.5"

# ——— Tokenizer & Model ———
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    ).eval()
# resize and tie_weights
model.resize_token_embeddings(len(tokenizer))
model.tie_weights()

# ——— Parameters ———
MASK_ID      = 126336                    # mask token 的 id
ANSWER_LEN   = 10                        # expected generation length
NUM_STEPS    = 10                        # reverse diffusion steps
GUIDANCE     = 1.0                       # classifier-free guidance scale

BLOCK_LEN    = ANSWER_LEN                

# Gumbel sample
TEMPERATURE  = 1.0
REMASK_STRAT = 'low_confidence'         # 'low_confidence' or 'random'

# ——— Prompt ———
prompt = "What is 2 + 2?"
print("Prompt:", prompt)

# ——— Tokenize & Move to Device ———
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids = inputs.input_ids  # shape: (1, L_prompt + ANSWER_LEN)

# ——— Generation ———
# generate(model, prompt_ids, steps, gen_length, block_length, 
#          temperature, cfg_scale, remasking, mask_id)

out = gen_diffusion(
    model,
    input_ids,
    NUM_STEPS,
    ANSWER_LEN,
    BLOCK_LEN,
    TEMPERATURE,
    GUIDANCE,
    REMASK_STRAT,
    MASK_ID,
)

# ——— Decode ———
gen_ids = out[0, input_ids.shape[1]:]  
answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
print("Answer:", answer)