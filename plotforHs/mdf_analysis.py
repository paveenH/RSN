#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 10:18:42 2025

@author: paveenhuang
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

    

TASKS = [
    "abstract_algebra",
    "anatomy",
    "econometrics",
    "global_facts",
    "jurisprudence"
]

model = "llama3"
size = "8B"
data_dir = os.path.join(os.getcwd(), "llama3_v3_mdf")
plot_dir = os.path.join(os.getcwd(), "plot", "llama3_v3_mdf")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

for task in TASKS:
    none_org_path = os.path.join(data_dir, f"none_{task}_{model}_{size}_org.npy")
    mdf_path = os.path.join(data_dir, f"none_{task}_{model}_{size}_mdf.npy")
    
    if not os.path.exists(none_org_path):
        print(f"[Error] Original file for {task} not found: {none_org_path}")
        continue
    if not os.path.exists(mdf_path):
        print(f"[Error] Modified file for {task} not found: {mdf_path}")
        continue

    none_org = np.load(none_org_path)      #  shape: (33, 4096)
    none_org = none_org[1:,]
    mdf_mean = np.load(mdf_path)           #  shape: (3, 31, 33, 4096)
    mdf_mean = mdf_mean[:,:,1:,:]

    print(f"\nTask: {task}")
    print("none_org shape:", none_org.shape)
    print("mdf_mean shape:", mdf_mean.shape)

    # For broadcasting, expand none_org to (1, 1, 33, 4096)
    none_org_expanded = none_org[np.newaxis, np.newaxis, :, :]

    # ================== 1. 计算归一化 L2 差异 ================== #
    diff = mdf_mean - none_org_expanded   # shape: (3, 31, 32, 4096)
    l2_diff = np.linalg.norm(diff, axis=-1)  # shape: (3, 31, 33)
    
    # 计算原始 none_org 的 L2 norm（用于归一化）
    layer_norms = np.linalg.norm(none_org, axis=-1, keepdims=True)  # shape: (32, 1)
    l2_diff_norm = l2_diff / layer_norms.T  # shape: (3, 31, 32)

    print("L2 diff shape:", l2_diff.shape)
    print("Normalized L2 diff shape:", l2_diff_norm.shape)

    # # 计算 impact factor
    # sum_l2_norm = np.sum(l2_diff_norm, axis=-1, keepdims=True)  # shape: (3, 31, 1)
    # impact_factor_l2 = l2_diff_norm / sum_l2_norm  # shape: (3, 31, 32)

    # For the last layer (index 32)
    # final_layer_impact_l2 = impact_factor_l2[:, :, 31]  # shape: (3, 31)
    final_layer_impact_l2 = l2_diff_norm[:, :, 31]  # shape: (3, 31)
    print("Final layer impact (L2 norm) shape:", final_layer_impact_l2.shape)
    print("Final layer impact (L2 norm) for task", task, ":\n", final_layer_impact_l2)

    # ================== 2. 计算 Cosine 相似度变化 ================== #
    # Cosine similarity 计算公式：
    # cos_sim = (A ⋅ B) / (||A|| * ||B||)
    dot_product = np.sum(mdf_mean * none_org_expanded, axis=-1)  # (3, 31, 32)
    norm_mdf = np.linalg.norm(mdf_mean, axis=-1)  # (3, 31, 32)
    norm_none = np.linalg.norm(none_org, axis=-1, keepdims=True).T  # (33,) -> (1, 1, 32)

    cos_sim = dot_product / (norm_mdf * norm_none)  # shape: (3, 31, 32)
    cos_change = 1 - cos_sim  # 计算 Cosine 变化（1-相似度）

    print("Cosine similarity shape:", cos_sim.shape)
    print("Cosine change shape:", cos_change.shape)

    # 关注最后一层（索引 32）的 Cosine 变化
    final_layer_cos_change = cos_change[:, :, 31]  # shape: (3, 31)
    print("Final layer impact (Cosine change) shape:", final_layer_cos_change.shape)
    print("Final layer impact (Cosine change) for task", task, ":\n", final_layer_cos_change)
    
    # ================== 3. 添加层间影响可视化 ================== #
    # 设置基本绘图参数
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['axes.edgecolor'] = '#333333'
    mpl.rcParams['xtick.major.width'] = 0.8
    mpl.rcParams['ytick.major.width'] = 0.8
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10

    # 定义TOP值和对应的标签
    top_values = [20, 640, 4096]
    top_labels = ["TOP-20", "TOP-640", "TOP-4096"]

    # 修改热力图代码，解决空行问题并从1开始计数
    # -------------- L2 差异的层间影响可视化（修正版） -------------- #
    # 先计算当前任务所有 TOP 值的全局最小和最大值
    global_min_l2 = np.min(l2_diff_norm)
    global_max_l2 = np.max(l2_diff_norm)

    for i, (top_val, top_label) in enumerate(zip(top_values, top_labels)):
        plt.figure(figsize=(12, 4))
        im_data = l2_diff_norm[i].T
        # 创建热力图，注意调整对齐方式
        im = plt.imshow(im_data, aspect='auto', cmap='viridis', 
                    interpolation='nearest', origin='upper',
                    vmin=global_min_l2, vmax=global_max_l2)
        
        # 设置刻度和网格 - 从1开始计数
        modified_layer_ticks = np.arange(0, 31, 5)  # 修改层的位置
        modified_layer_labels = [str(i+1) for i in modified_layer_ticks]  # 从1开始的标签
        
        affected_layer_ticks = np.arange(0, 32, 4)  # 受影响层的位置
        affected_layer_labels = [str(i+1) for i in affected_layer_ticks]  # 从1开始的标签
        
        plt.xticks(modified_layer_ticks, modified_layer_labels)
        plt.yticks(affected_layer_ticks, affected_layer_labels)
        
        # 添加网格线
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, 31, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 32, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.2)
        
        # 设置标签和标题
        plt.xlabel("Modified Layer Index", fontsize=11, fontweight='medium')
        plt.ylabel("Affected Layer Index", fontsize=11, fontweight='medium')
        task_name = task.replace("_", " ")
        plt.title(f"Layer-wise Impact (Normalized L2) - {top_label} for {task_name}", 
                  fontsize=12, fontweight='bold', pad=10)
        
        # 添加颜色条
        cbar = plt.colorbar(im, pad=0.01)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label('Normalized L2 Impact', size=10)
        cbar.outline.set_linewidth(0.5)
        
        # 设置轴的范围，确保完整显示所有数据点
        plt.xlim(-0.5, 30.5)
        plt.ylim(-0.5, 31.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{task}_layer_impact_l2_{top_val}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.show()

    # -------------- Cosine 变化的层间影响可视化（修正版） -------------- #
    global_min_cos = np.min(cos_change)
    global_max_cos = np.max(cos_change)

    for i, (top_val, top_label) in enumerate(zip(top_values, top_labels)):
        plt.figure(figsize=(12, 4))
        
        # 创建热力图，使用origin='upper'避免空行
        im_data_cos = cos_change[i].T
        im = plt.imshow(im_data_cos, aspect='auto', cmap='plasma', 
                    interpolation='nearest', origin='upper',
                    vmin=global_min_cos, vmax=global_max_cos)
        
        # 设置刻度和网格 - 从1开始计数
        modified_layer_ticks = np.arange(0, 31, 5)
        modified_layer_labels = [str(i+1) for i in modified_layer_ticks]
        
        affected_layer_ticks = np.arange(0, 32, 5)
        affected_layer_labels = [str(i+1) for i in affected_layer_ticks]
        
        plt.xticks(modified_layer_ticks, modified_layer_labels)
        plt.yticks(affected_layer_ticks, affected_layer_labels)
        
        # 添加网格线
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, 31, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 32, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.2)
        
        # 设置标签和标题
        plt.xlabel("Modified Layer Index", fontsize=11, fontweight='medium')
        plt.ylabel("Affected Layer Index", fontsize=11, fontweight='medium')
        task_name = task.replace("_", " ")
        plt.title(f"Layer-wise Impact (Cosine Change) - {top_label} for {task_name}", 
                  fontsize=12, fontweight='bold', pad=10)
        
        # 添加颜色条
        cbar = plt.colorbar(im, pad=0.01)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label('Cosine Change', size=10)
        cbar.outline.set_linewidth(0.5)
        
        # 设置轴的范围
        plt.xlim(-0.5, 30.5)
        plt.ylim(-0.5, 31.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{task}_layer_impact_cos_{top_val}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.show()

    # 组合图也需要修改为从1开始计数
    # -------------- 组合图：每个TOP值的最后一层影响（修正版） -------------- #
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o-', 's-', '^-']

    # x轴坐标从1开始
    x_coords = np.arange(1, 32)  # 1到31

    for i, (top_val, top_label) in enumerate(zip(top_values, top_labels)):
        plt.plot(x_coords, final_layer_impact_l2[i], markers[i], 
                 color=colors[i], label=top_label, linewidth=1.5, markersize=5)

    plt.xlabel("Modified Layer Index", fontsize=11, fontweight='medium')
    plt.ylabel("Impact on Final Layer (L2 Norm)", fontsize=11, fontweight='medium')
    task_name = task.replace("_", " ")
    plt.title(f"Impact on Final Layer by Different TOP Values - {task_name}", 
              fontsize=12, fontweight='bold', pad=10)

    # 设置刻度，从1开始
    plt.xticks(np.arange(1, 32, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    plt.savefig(os.path.join(plot_dir, f"{task}_combined_impact_l2.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()

    # Cosine变化组合图
    plt.figure(figsize=(10, 6))

    for i, (top_val, top_label) in enumerate(zip(top_values, top_labels)):
        plt.plot(x_coords, final_layer_cos_change[i], markers[i], 
                 color=colors[i], label=top_label, linewidth=1.5, markersize=5)

    plt.xlabel("Modified Layer Index", fontsize=11, fontweight='medium')
    plt.ylabel("Impact on Final Layer (Cosine Change)", fontsize=11, fontweight='medium')
    task_name = task.replace("_", " ")
    plt.title(f"Impact on Final Layer by Different TOP Values - {task_name}", 
              fontsize=12, fontweight='bold', pad=10)

    # 设置刻度，从1开始
    plt.xticks(np.arange(1, 32, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, f"{task}_combined_impact_cos.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()


