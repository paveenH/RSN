#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:17:31 2024

@author: paveenhuang
"""

import os
import glob
import pandas as pd
from omegaconf import OmegaConf

LOGS_DIR = os.path.join(os.getcwd(), "/logs/eval/runs/")
COLLECTED_DIR = os.path.join(os.getcwd(), "/logs/collected_metrics")
os.makedirs(COLLECTED_DIR, exist_ok=True)

run_dirs = glob.glob(os.path.join(LOGS_DIR, "*"))
for run_dir in run_dirs:
    # Define the path to config.yaml
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.isfile(config_path):
        print(f"The configuration file does not exist: {config_path}")
        continue
    
    try:
        cfg = OmegaConf.load(config_path)
        task_name = cfg.get("data.dataset_partial.task", "unknown_task")
    except Exception as e:
        print(f"Failed to parse the configuration file: {config_path}, erroe: {e}")
        continue
    
    metrics_path = os.path.join(run_dir, "csv", "version_0", "metrics.csv")
    if not os.path.isfile(metrics_path):
        print(f"metrics.csv does not exist: {metrics_path}")
        continue
    
    try:
        metrics_df = pd.read_csv(metrics_path)
    except Exception as e:
        print(f"Failed to read metrics.csv: {metrics_path}, error: {e}")
        continue
    
    # Define the saved CSV file path
    collected_metrics_path = os.path.join(COLLECTED_DIR, f"{task_name}.csv")
    
    try:
        metrics_df.to_csv(collected_metrics_path, index=False)
        print(f"save: {collected_metrics_path}")
    except Exception as e:
        print(f"save {collected_metrics_path} failed, error: {e}")