import os
import json
import numpy as np
from transformers import AutoTokenizer

# ---------------- Config ---------------- #

MODEL = "llama3"
SIZE = "8B"
TYPE = "non"
DIR = "/data2/paveen/RolePlaying/components"
json_dir = os.path.join(DIR, f"answer_{TYPE}_logits", MODEL)

# Choose TOKEN mode or CHAR mode
USE_TOKENS = False  # set True if you want token length
tokenizer = None
if USE_TOKENS:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")  # or any tokenizer you use


# ---------------- Helper ---------------- #

def get_prompt_length(text):
    if USE_TOKENS:
        return len(tokenizer.encode(text))
    return len(text)  # character length


def extract_lengths_from_json(json_path, task):
    """Return lists of lengths: all_samples_lengths, divergent_samples_lengths."""
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
        data = obj["data"]

    all_lengths = []
    div_lengths = []

    for entry in data:
        prompt = entry.get("text")  # change if prompt key differs
        L = get_prompt_length(prompt)
        all_lengths.append(L)

        # divergent test
        ans_non = entry.get(f"answer_non_{task}")
        ans_exp = entry.get(f"answer_{task}")

        if ans_non != ans_exp:
            div_lengths.append(L)

    return all_lengths, div_lengths


# ---------------- Main ---------------- #

all_samples = []
div_samples = []

for filename in os.listdir(json_dir):
    if filename.endswith(f"_{SIZE}_answers.json"):
        # Extract task name:  taskname_size_answers.json
        task = filename.replace(f"_{SIZE}_answers.json", "")
        task = task.replace("_", " ")  # recover original task name (for answer keys)

        json_path = os.path.join(json_dir, filename)

        lengths_all, lengths_div = extract_lengths_from_json(json_path, task)

        all_samples.extend(lengths_all)
        div_samples.extend(lengths_div)

# ---------------- Stats ---------------- #

all_mean = np.mean(all_samples)
all_std = np.std(all_samples)

div_mean = np.mean(div_samples)
div_std = np.std(div_samples)

print("\n======== Prompt Length Statistics ========")
print(f"All Samples     : mean={all_mean:.2f}, std={all_std:.2f}, n={len(all_samples)}")
print(f"Divergent Samples : mean={div_mean:.2f}, std={div_std:.2f}, n={len(div_samples)}")


