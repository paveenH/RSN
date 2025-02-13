#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:00:42 2025

@author: paveenhuang
"""

import os
import re
import json
import copy


model = "llama3_v3"  
# alpha = 5
# data_dir = os.path.join(os.getcwd(), f"{model}/answer_modified_alpha{alpha}")
# save_dir = os.path.join(os.getcwd(), f"{model}/answer_modified_alpha{alpha}_revised")
data_dir = os.path.join(os.getcwd(), f"{model}/answer_modified")
save_dir = os.path.join(os.getcwd(), f"{model}/answer_modified_revised")

os.makedirs(save_dir, exist_ok=True)

# pattern = re.compile(r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3.8B|7B|8B)_answers\.json$")
pattern = re.compile(r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3.8B|7B|8B)_answers(_\d+)?\.json$")

def cleaning(generated_output):
    """
    Clean the generated output to extract the answer option (A, B, C, D).
    Uses regular expressions to find the first occurrence of A), B), C), or D) and returns the corresponding letter.
    """
    match = re.search(r'\b([A-E])\b', generated_output.upper())
    if match:
        return match.group(1)
    else:
        return generated_output.strip().upper()
    
    
def get_correct_answer(label):
    """
    Map label integer to correct answer letter.
    Assuming label is 1-based: 1='A', 2='B', 3='C', 4='D', 5='E'
    """
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    return mapping.get(label, None)

for file in os.listdir(data_dir):
    if "_answers" in file:
        match_file = pattern.match(file)
        if not match_file:
            print(f"Filename '{file}' does not match the expected pattern. Skipping.")
            continue

        task = match_file.group("task")
        size = match_file.group("size")
        file_path = os.path.join(data_dir, file)
        save_file_path = os.path.join(save_dir, file)

        # Read the JSON file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file}: {e}")
            continue

        if "accuracy" in data:
            data["original_accuracy"] = copy.deepcopy(data["accuracy"])

        # For convenience, define some variables that map your label chars
        valid_answers = ["A", "B", "C", "D", "E"]

        # Replace underscores with spaces for the display name
        task_name = task.replace('_', ' ')
        character_keys = [f"none {task_name}",f"{task_name}"]

        # A little helper to safely get stats from data["accuracy"]
        def get_stats(char_key):
            if "accuracy" not in data:
                return None
            return data["accuracy"].get(char_key, None)

        # Function to adjust stats
        def adjust_stats(stats_dict, new_status=None):
            """
            Adjusts the stats dictionary by incrementing new_status
            """
            if stats_dict is None:
                return

            if new_status and new_status in stats_dict:
                stats_dict[new_status] += 1

            # Recompute accuracy_percentage = (correct / total) * 100
            correct = stats_dict.get("correct", 0)
            total = stats_dict.get("total", 0)
            if total > 0:
                stats_dict["accuracy_percentage"] = round((correct / total) * 100, 2)
            else:
                stats_dict["accuracy_percentage"] = 0.0
                
        # Process each sample
        for idx, sample in enumerate(data["data"]):
            label_int = sample.get("label", None)
            correct_answer = get_correct_answer(label_int)
            
            for key in list(sample.keys()):
                if not key.startswith("answer_"):
                    continue
                
                original_answer = sample[key]                    
                
                if "[Add]" in original_answer:
                    original_key = key + "_extract"
                    sample[original_key] = original_answer
                    sample[key] = correct_answer
                    continue
                
                if original_answer in valid_answers:
                    continue

                cleaned_ans = cleaning(original_answer)
                char_str = key.replace("answer_", "").replace("_", " ")   
                stats = get_stats(char_str)
                original_key = key + "_original"
                
                if cleaned_ans in valid_answers:            
                    stats ["invalid"] -= 1 
                    # Determine new_status based on whether the cleaned answer matches the correct answer
                    if cleaned_ans == correct_answer:
                        new_status = "correct"
                    elif cleaned_ans == "E" :
                        new_status = "E_count"
                    else:
                        new_status = ""

                    # adjust stats
                    if stats and new_status:
                        adjust_stats(stats, new_status=new_status)
                    
                    # Save original answer in a new key
                    sample[original_key] = original_answer
                    sample[key] = cleaned_ans
                    print(f"Revised: {original_answer} -> {cleaned_ans}, sample #{idx}, key={key}")
                
                elif "i am not sure" in original_answer.lower():
                    stats["invalid"] -= 1
                    new_status = "E_count"
                    adjust_stats(stats, new_status=new_status)
                    sample[original_key] = original_answer
                    sample[key] = "E"
                    print(f"Revised (I am not sure fallback): {original_answer} -> E, sample #{idx}, key={key}")                    
                        

        with open(save_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Processed and updated file: {file_path}")
                
            
            
            