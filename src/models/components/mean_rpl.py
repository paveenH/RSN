#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 10:28:16 2025

Author: paveenhuang

This script:
1) Iterates over start-end pairs [0..31).
2) Loads the mean hidden states from each .npy file (which has shape (1,1,32,hidden_size)).
3) Removes extra dims -> shape (32, hidden_size).
4) Stacks them into a single array with shape (31, 32, hidden_size).
5) Saves the final stacked array as one .npy file.

So effectively, for each task, we get a single matrix of shape 31 x 32 x hidden_size.
"""

import os
import numpy as np

# 1) 定义任务、模型、大小
TASKS = [
    "abstract_algebra",
    "anatomy",
    "global_facts",
    "econometrics",
    "jurisprudence"
]
MODEL = "llama3"
SIZE = "8B"

# 2) 目录配置
INPUT_DIR = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v3_rpl/{MODEL}"
OUTPUT_DIR = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v3_rpl_mean/{MODEL}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3) 生成 start-end 对: (0,1), (1,2), ..., (30,31)
START = 0
END = 31
START_END_PAIRS = [(i, i+1) for i in range(START, END)]

for task in TASKS:
    # 用来收集所有 (start, end) 的 mean hidden states
    stacked_list = []

    for (s, e) in START_END_PAIRS:
        # 读取单层替换后的均值文件
        # 例如： "abstract_algebra_8B_0_1.npy"
        mean_file = os.path.join(INPUT_DIR, f"{task}_{SIZE}_{s}_{e}.npy")

        if not os.path.isfile(mean_file):
            print(f"File not found: {mean_file}, skipping.")
            continue

        # 加载：形状 (1,1,32,hidden_size)
        hs_mean = np.load(mean_file)
        print(f"Loaded {mean_file} with shape={hs_mean.shape}")

        # 去掉前两个维度 -> (32, hidden_size)
        # 原先 shape(1,1,32,hidden_size)
        #   索引 [0, 0] 后剩 (32, hidden_size)
        layer_hs = hs_mean[0, 0]
        print(f"After squeezing, shape={layer_hs.shape}")

        stacked_list.append(layer_hs)

    if not stacked_list:
        print(f"No data found for task={task}, skipping final stack.")
        continue

    # 堆叠 -> (num_replaced, 32, hidden_size)
    # 若所有层都存在，则 num_replaced ~ 31
    # shape => (31, 32, hidden_size)
    big_arr = np.stack(stacked_list, axis=0)
    print(f"Final stacked shape for {task}: {big_arr.shape}")

    # 4) 保存合并后的大文件
    # 例如： "abstract_algebra_8B_alllayers.npy"
    out_file = os.path.join(OUTPUT_DIR, f"{task}_{SIZE}_alllayers.npy")
    np.save(out_file, big_arr)
    print(f"Saved stacked mean HS to: {out_file}")

print("All tasks completed!")