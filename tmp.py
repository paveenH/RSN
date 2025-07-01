#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 12:30:54 2025

@author: paveenhuang
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

text = "non abstract algebra expert"

tokens = tokenizer.tokenize(text)

print(tokens)