import os
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

# ---------------- Config ---------------- #

MODEL = "llama3"
SIZE = "8B"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TYPE = "non"
DIR = "/data2/paveen/RolePlaying/components"
json_dir = os.path.join(DIR, f"answer_{TYPE}_logits", MODEL)

USE_TOKENS = False
tokenizer = None
if USE_TOKENS:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ---------------- Helper functions ---------------- #

def get_prompt_length(text):
    if text is None:
        return 0
    if USE_TOKENS:
        return len(tokenizer.encode(text))
    return len(text)


def normalize_task_name(task):
    return task.replace(" ", "_")


def load_lengths_for_task(task):
    """Return: total_lengths, divergent_lengths"""
    norm_task = normalize_task_name(task)   # convert "electrical engineering" → "electrical_engineering"
    json_path = os.path.join(json_dir, f"{norm_task}_{SIZE}_answers.json")

    if not os.path.exists(json_path):
        print(f"[WARN] Missing: {json_path}")
        return [], []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    total_lengths = []
    divergent_lengths = []

    for entry in data:
        # detect prompt field
        prompt = entry.get("prompt") or entry.get("text") or entry.get("input") or ""
        L = get_prompt_length(prompt)
        total_lengths.append(L)

        # JSON uses underscore naming
        ans_non = entry.get(f"answer_non_{norm_task}")
        ans_exp = entry.get(f"answer_{norm_task}")

        if ans_non != ans_exp:
            divergent_lengths.append(L)

    print(f"[Task {norm_task}] Inconsistent: {len(divergent_lengths)}  /  Total: {len(total_lengths)}")

    return total_lengths, divergent_lengths


# ---------------- Domains ---------------- #

stem_tasks = [
    "abstract algebra",
    "anatomy",
    "astronomy",
    "college biology",
    "college chemistry",
    "college computer science",
    "college mathematics",
    "college physics",
    "computer security",
    "conceptual physics",
    "electrical engineering",
    "elementary mathematics",
    "high school biology",
    "high school chemistry",
    "high school computer science",
    "high school mathematics",
    "high school physics",
    "high school statistics",
    "machine learning",
]

humanities_tasks = [
    "formal logic",
    "high school european history",
    "high school us history",
    "high school world history",
    "international law",
    "jurisprudence",
    "logical fallacies",
    "moral disputes",
    "moral scenarios",
    "philosophy",
    "prehistory",
    "professional law",
    "world religions",
]

social_sciences_tasks = [
    "econometrics",
    "high school geography",
    "high school government and politics",
    "high school macroeconomics",
    "high school microeconomics",
    "high school psychology",
    "human sexuality",
    "professional psychology",
    "public relations",
    "security studies",
    "sociology",
    "us foreign policy",
]

other_tasks = [
    "business ethics",
    "clinical knowledge",
    "college medicine",
    "global facts",
    "human aging",
    "management",
    "marketing",
    "medical genetics",
    "miscellaneous",
    "nutrition",
    "professional accounting",
    "professional medicine",
    "virology",
]

categories_map = {
    "STEM": stem_tasks,
    "Humanities": humanities_tasks,
    "Social Sciences": social_sciences_tasks,
    "Other": other_tasks,
}


# ---------------- Main: compute domain statistics ---------------- #

rows = []

for domain, task_list in categories_map.items():
    domain_total_lengths = []
    domain_div_lengths = []
    domain_total_count = 0
    domain_div_count = 0

    for task in task_list:
        t_total, t_div = load_lengths_for_task(task)
        domain_total_lengths.extend(t_total)
        domain_div_lengths.extend(t_div)
        domain_total_count += len(t_total)
        domain_div_count += len(t_div)

    total_mean = np.mean(domain_total_lengths) if domain_total_lengths else 0
    total_std  = np.std(domain_total_lengths) if domain_total_lengths else 0

    div_mean = np.mean(domain_div_lengths) if domain_div_lengths else 0
    div_std  = np.std(domain_div_lengths) if domain_div_lengths else 0

    rows.append({
        "Domain": domain,
        "Total Samples": domain_total_count,
        "Divergent Samples": domain_div_count,
        "Total Avg Length": round(total_mean,2),
        "Total Std": round(total_std,2),
        "Divergent Avg Length": round(div_mean,2),
        "Divergent Std": round(div_std,2),
        "% Divergent": round(domain_div_count / domain_total_count * 100, 2) if domain_total_count else 0
    })

# overall
all_total_lengths = []
all_div_lengths = []
for task in [t for ts in categories_map.values() for t in ts]:
    t_total, t_div = load_lengths_for_task(task)
    all_total_lengths.extend(t_total)
    all_div_lengths.extend(t_div)

rows.append({
    "Domain": "Overall",
    "Total Samples": len(all_total_lengths),
    "Divergent Samples": len(all_div_lengths),
    "Total Avg Length": round(np.mean(all_total_lengths), 2),
    "Total Std": round(np.std(all_total_lengths), 2),
    "Divergent Avg Length": round(np.mean(all_div_lengths), 2),
    "Divergent Std": round(np.std(all_div_lengths), 2),
    "% Divergent": round(len(all_div_lengths) / len(all_total_lengths) * 100, 2)
})

df = pd.DataFrame(rows)
print(df)