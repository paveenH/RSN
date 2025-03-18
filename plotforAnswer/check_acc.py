import os
import json
import re
from collections import defaultdict

# =======================
# Configuration
# =======================
model = "llama3_V3"  
answer_name = "answer_honest_revised"

data_dir = os.path.join(os.getcwd(), f"{model}/{answer_name}")
output_dir = os.path.join(os.getcwd(), f"{model}/counts")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize a nested defaultdict to store detailed metrics per task and size
# Structure: report_data[task][size] = {metrics}
report_data = defaultdict(lambda: defaultdict(lambda: {
    "elementary_mat_correct": 0,
    "elementary_mat_E_count": 0,
    "elementary_mat_invalid": 0,
    "none_elementary_mat_correct": 0,
    "none_elementary_mat_E_count": 0,
    "none_elementary_mat_invalid": 0,
    "elementary_mat_total": 0,
    "none_elementary_mat_total": 0
}))

# This regex pattern matches filenames like "task_7B_answers.json"
# pattern = re.compile(r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3\.8B|7B|8B)_answers\.json$")
pattern = re.compile(r"^(?P<task>.+)_(?P<size>1B|3B|3\.8B|7B|8B)_answers\.json$")


# Mapping from label index to answer letter
label_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

# =======================
# Processing JSON Files
# =======================
for file in os.listdir(data_dir):
    if file.endswith("_answers.json"):
        match = pattern.match(file)
        if match:
            task = match.group("task")  # For example "college chemistry"
            size = match.group("size")  # For example "7B"
            file_path = os.path.join(data_dir, file)
            
            # Dynamically create field names based on the task
            answer_field = f"answer_{task}"
            answer_none_field = f"answer_none_{task}"
            
            # Read the JSON file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error reading file {file}: {e}")
                continue
            
            # Get the accuracy field from the JSON
            accuracy_dict = data.get("accuracy", {})
            elem_accuracy_key = task.replace("_", " ")
            none_elem_accuracy_key = f"none {elem_accuracy_key}"
            
            # Get the provided accuracy statistics
            provided_accuracy_elem = accuracy_dict.get(elem_accuracy_key, {})
            provided_accuracy_none_elem = accuracy_dict.get(none_elem_accuracy_key, {})
            
            # Process each data entry
            entries = data.get("data", [])
            for entry in entries:
                # Process Elementary Mathematics (or the current task)
                label = entry.get("label")
                correct_answer = label_to_letter.get(label, None)
                model_answer_elem = entry.get(answer_field, "").strip()
                
                # Check if the model's answer matches the correct one
                if model_answer_elem == correct_answer:
                    report_data[task][size]["elementary_mat_correct"] += 1
                elif model_answer_elem == "E":
                    report_data[task][size]["elementary_mat_E_count"] += 1
                elif model_answer_elem not in ["A", "B", "C", "D", "E"]:
                    report_data[task][size]["elementary_mat_invalid"] += 1
                
                report_data[task][size]["elementary_mat_total"] += 1
                
                # Process None Elementary Mathematics (or task outside of the main task)
                model_answer_none_elem = entry.get(answer_none_field, "").strip()
                
                # Assuming "E" is the correct answer for the "None" category
                if model_answer_none_elem == correct_answer:
                    report_data[task][size]["none_elementary_mat_correct"] += 1
                elif model_answer_none_elem == "E":
                    report_data[task][size]["none_elementary_mat_E_count"] += 1
                elif model_answer_none_elem not in ["A", "B", "C", "D", "E"]:
                   report_data[task][size]["none_elementary_mat_invalid"] += 1
                
                report_data[task][size]["none_elementary_mat_total"] += 1
            
            # Retrieve provided accuracy statistics for Elementary Mathematics
            provided_correct_elem = provided_accuracy_elem.get("correct", "N/A")
            provided_E_count_elem = provided_accuracy_elem.get("E_count", "N/A")
            provided_invalid_elem = provided_accuracy_elem.get("invalid", "N/A")
            provided_accuracy_percentage_elem = provided_accuracy_elem.get("accuracy_percentage", "N/A")
            
            # Retrieve provided accuracy statistics for None Elementary Mathematics
            provided_correct_none = provided_accuracy_none_elem.get("correct", "N/A")
            provided_E_count_none = provided_accuracy_none_elem.get("E_count", "N/A")
            provided_invalid_none = provided_accuracy_none_elem.get("invalid", "N/A")
            provided_accuracy_percentage_none = provided_accuracy_none_elem.get("accuracy_percentage", "N/A")
            
            # Calculate actual accuracy for Elementary Mathematics
            computed_elem_correct = report_data[task][size]["elementary_mat_correct"]
            computed_elem_E_count = report_data[task][size]["elementary_mat_E_count"]
            computed_elem_invalid = report_data[task][size]["elementary_mat_invalid"]
            computed_elem_total = report_data[task][size]["elementary_mat_total"]
            computed_elem_accuracy = (computed_elem_correct / computed_elem_total) * 100 if computed_elem_total else 0.0
            
            computed_none_correct = report_data[task][size]["none_elementary_mat_correct"]
            computed_none_E_count = report_data[task][size]["none_elementary_mat_E_count"]
            computed_none_invalid = report_data[task][size]["none_elementary_mat_invalid"]
            computed_none_total = report_data[task][size]["none_elementary_mat_total"]
            computed_none_accuracy = (computed_none_correct / computed_none_total) * 100 if computed_none_total else 0.0
            
            # Print only mismatched accuracy percentages
            if (abs(computed_elem_accuracy - float(provided_accuracy_percentage_elem if provided_accuracy_percentage_elem != "N/A" else computed_elem_accuracy)) > 0.01 or
                computed_elem_E_count != (int(provided_E_count_elem) if provided_E_count_elem != "N/A" else computed_elem_E_count)):
                print(f"\n=== File: {file} ===")
                print(f"Task: {task}, Size: {size}")
                print("\n-- Elementary Mathematics (or corresponding task) --")
                print(f"Computed Correct: {computed_elem_correct}")
                print(f"Provided Correct: {provided_correct_elem}")
                print(f"Computed E_count: {computed_elem_E_count}")
                print(f"Provided E_count: {provided_E_count_elem}")
                print(f"Computed Invalid: {computed_elem_invalid}")
                print(f"Provided Invalid: {provided_invalid_elem}")
                print(f"Computed Total: {computed_elem_total}")
                print(f"Computed Accuracy Percentage: {computed_elem_accuracy:.2f}%")
                print(f"Provided Accuracy Percentage: {provided_accuracy_percentage_elem}%")

            if (abs(computed_none_accuracy - float(provided_accuracy_percentage_none if provided_accuracy_percentage_none != "N/A" else computed_none_accuracy)) > 0.01 or
                computed_none_E_count != (int(provided_E_count_none) if provided_E_count_none != "N/A" else computed_none_E_count)):
                print(f"\n=== File: {file} ===")
                print(f"Task: {task}, Size: {size}")
                print("\n-- None Elementary Mathematics (or task outside the main task) --")
                print(f"Computed Correct: {computed_none_correct}")
                print(f"Provided Correct: {provided_correct_none}")
                print(f"Computed E_count: {computed_none_E_count}")
                print(f"Provided E_count: {provided_E_count_none}")
                print(f"Computed Invalid: {computed_none_invalid}")
                print(f"Provided Invalid: {provided_invalid_none}")
                print(f"Computed Total: {computed_none_total}")
                print(f"Computed Accuracy Percentage: {computed_none_accuracy:.2f}%")
                print(f"Provided Accuracy Percentage: {provided_accuracy_percentage_none}%")



