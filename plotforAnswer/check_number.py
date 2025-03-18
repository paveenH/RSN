#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:21:07 2025

@author: paveenhuang
"""
import os
import re
from collections import defaultdict

model = "llama3_v3"  
answer_name = "answer_modified_layer_revised"
size = "8B"
top = 20
start = 0
end = 31

data_dir = os.path.join(os.getcwd(), f"{model}/{answer_name}")
pattern = re.compile(
    r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3\.8B|7B|8B)_answers"
    r"(?:_(?P<top>\d+))?"
    r"(?:_(?P<start>\d+)_(?P<end>\d+))?"
    r"\.json$"
)

TASKS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_medicine",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)

# Use a dictionary to store the task set corresponding to each size
size_tasks = defaultdict(set)

for file in os.listdir(data_dir):
    if f"{size}_answers_{top}_{start}_{end}" in file:
        match = pattern.match(file)
        if match:
            task = match.group("task")
            size = match.group("size")
            size_tasks[size].add(task)

# For each size, if the number of tasks is less than 57, list the missing tasks
print("Missing tasks for sizes with less than 57 files:")
for size, tasks in size_tasks.items():
    if len(tasks) < len(TASKS):
        missing = set(TASKS) - tasks
        print(f"\nSize {size} has {len(tasks)} files. Missing tasks:")
        for t in sorted(missing):
            print("  -", t)
    else:
        print(f"\nSize {size} has all {len(TASKS)} tasks.")