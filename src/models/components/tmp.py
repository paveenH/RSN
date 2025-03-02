#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 21:37:45 2025

@author: paveenhuang
"""

import os

# 目标目录（请修改为你的目录）
directory = "/data2/paveen/RolePlaying/src/models/components/hidden_states_modified/llama3"

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if "4960" in filename:
        # 生成新的文件名
        new_filename = filename.replace("4960", "4096")
        
        # 构造完整路径
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

print("All matching files have been renamed.")