#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:43:04 2025

@author: paveenhuang
"""

from datasets import load_dataset

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation", cache_dir="/data2/paveen/RolePlaying/.cache")
print(ds)
print(ds[0])