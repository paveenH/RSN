#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 14:22:38 2025

@author: paveenhuang
"""

pip install -U lm-eval

lm_eval --tasks list | grep truthfulqa

lm_eval \
  --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=float16,device_map=auto \
  --tasks truthfulqa_mc1,truthfulqa_mc2 \
  --batch_size auto \
  --num_fewshot 0 \
  --output_path results/original_llama3_8b_tqa.json