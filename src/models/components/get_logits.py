#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 22:20:52 2025

@author: paveenhuang
"""

import argparse
import json
import os
import numpy as np
from vicuna import VicundaModel

# Label mapping: A, B, C, D 对应 index 0,1,2,3
LABEL_MAPPING = ["A", "B", "C", "D"]

def parse_arguments_and_define_characters():
    """
    解析命令行参数，返回 task, model, size
    并定义需要测试的角色列表。
    """
    parser = argparse.ArgumentParser(description="Extract logits for each role")
    parser.add_argument("task_size", type=str, help="The task, model, and size as a combined argument.")
    args = parser.parse_args()
    try:
        task, model, size = args.task_size.split()
    except ValueError:
        raise ValueError("The task_size parameter should contain three parts: task, model, and size.")
    
    # 定义角色列表（根据你的需求修改）
    characters = [f"none {task} expert", f"{task} student", f"{task} expert", "person"]
    return task, model, size, characters

def load_json_data(json_path):
    print(f"Loading JSON data from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total samples loaded: {len(data)}")
    return data

def get_option_token_ids(vc):
    """
    使用模型的 tokenizer 获取选项 "A", "B", "C", "D" 对应的 token id（假定单字 token）。
    """
    option_token_ids = []
    for option in LABEL_MAPPING:
        # 注意：此处假设每个选项编码后只有一个 token
        token_ids = vc.tokenizer.encode(option, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Option {option} does not map to exactly one token: {token_ids}")
        option_token_ids.append(token_ids[0])
    return option_token_ids

def compute_softmax(logits):
    """计算 softmax 得到概率分布"""
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()

def main():
    # 解析参数
    task, model_name, size, roles = parse_arguments_and_define_characters()
    
    # 定义路径
    # 假定 mmlu 数据存放在此处，文件名为 "{task}.json"
    data_path = os.path.join("/data2/paveen/RolePlaying/src/models/components/mmlu", f"{task}.json")
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    save_dir = "/data2/paveen/RolePlaying/src/models/components/logits_v3_4ops"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化模型，建议多卡加载（此处根据你的环境设置，如使用 CUDA_VISIBLE_DEVICES=...）
    vc = VicundaModel(model_path=model_path, num_gpus=2)
    
    # 加载数据
    data = load_json_data(data_path)
    
    # 获取选项对应的 token ids
    option_token_ids = get_option_token_ids(vc)
    
    # 用于存储每个角色的正确预测样本中对应正确选项的 logits 或概率
    results = {role: [] for role in roles}
    
    # 遍历数据集（你也可以限制样本数量）
    for idx, sample in enumerate(data):
        context = sample.get("text", "")
        true_label_int = sample.get("label", -1)
        if true_label_int < 0 or true_label_int >= len(LABEL_MAPPING):
            print(f"Sample {idx} invalid label {true_label_int}; skipping.")
            continue
        true_label = LABEL_MAPPING[true_label_int]  # 比如 "C"
        
        # 对于每个角色，构造对应的 prompt，并获取 logits
        for role in roles:
            # 使用模型的 template 构造 prompt
            prompt = vc.template.format(character=role, context=context)
            
            # 调用 get_logits 得到 logits
            # 这里假设 get_logits 返回形状 [batch_size, seq_len, vocab_size]
            logits = vc.get_logits([prompt], character=role)  # logits: tensor, shape (1, seq_len, vocab_size)
            
            # 取最后一个 token 的 logits作为预测依据
            # 注意：此处需要将 logits 转为 numpy 数组（确保在 CPU 上）
            last_logits = logits[0, -1, :].detach().cpu().numpy()
            
            # 提取选项 "A","B","C","D" 对应的 logits
            option_logits = [last_logits[tid] for tid in option_token_ids]
            
            # 得到 softmax 概率（或直接用原始 logits 比较）
            option_probs = compute_softmax(np.array(option_logits))
            
            # 预测的选项为概率最高的那个
            pred_idx = int(np.argmax(option_probs))
            pred_label = LABEL_MAPPING[pred_idx]
            
            # 如果预测正确，则将该样本的正确选项的 logit 和概率保存下来
            if pred_label == true_label:
                # 保存一组数据，可以保存原始 logit 和 softmax 概率
                results[role].append({
                    "sample_idx": idx,
                    "true_label": true_label,
                    "predicted_logit": option_logits[true_label_int],
                    "predicted_prob": option_probs[true_label_int],
                    "all_option_logits": option_logits,
                    "all_option_probs": option_probs,
                })
    
    # 输出结果：计算每个角色在正确预测样本中平均的正确选项的 logit 和概率
    summary = {}
    for role, vals in results.items():
        if vals:
            avg_logit = np.mean([v["predicted_logit"] for v in vals])
            avg_prob = np.mean([v["predicted_prob"] for v in vals])
            summary[role] = {"avg_logit": avg_logit, "avg_prob": avg_prob, "n": len(vals)}
        else:
            summary[role] = {"avg_logit": None, "avg_prob": None, "n": 0}
    
    print("Summary of logits on correctly predicted samples per role:")
    for role, info in summary.items():
        print(f"Role: {role}  Samples: {info['n']}  Avg Logit: {info['avg_logit']}, Avg Prob: {info['avg_prob']}")
    
    # 保存详细结果和 summary 到 JSON 文件中
    output = {"detailed": results, "summary": summary}
    out_path = os.path.join(save_dir, f"logits_{task}_{model_name}_{size}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()