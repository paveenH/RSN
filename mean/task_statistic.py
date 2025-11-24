import os
import json
import pandas as pd

# ---------------- Domain definitions ---------------- #

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


# ---------------- Config ---------------- #

MODEL = "llama3"
SIZE = "8B"
TYPE = "non"

DIR = "/data2/paveen/RolePlaying/components"
json_dir = os.path.join(DIR, f"answer_{TYPE}_logits", MODEL)


# ---------------- Helper: load divergent count for one task ---------------- #

def count_divergent(task):
    json_path = os.path.join(json_dir, f"{task}_{SIZE}_answers.json")
    if not os.path.exists(json_path):
        print(f"[WARN] Missing JSON: {json_path}")
        return 0, 0  # total, divergent

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    total = len(data)
    divergent = 0

    for entry in data:
        ans_non = entry.get(f"answer_non_{task}")
        ans_exp = entry.get(f"answer_{task}")
        if ans_non != ans_exp:
            divergent += 1

    return total, divergent


# ---------------- Main aggregation ---------------- #

results = []

for domain, tasks in categories_map.items():
    domain_total = 0
    domain_div = 0

    for task in tasks:
        t_total, t_div = count_divergent(task)
        domain_total += t_total
        domain_div += t_div

    div_ratio = (domain_div / domain_total * 100) if domain_total > 0 else 0
    
    results.append({
        "Domain": domain,
        "Total Samples": domain_total,
        "Divergent Samples": domain_div,
        "% Divergent": round(div_ratio, 2),
    })

# Compute overall
overall_total = sum(r["Total Samples"] for r in results)
overall_div = sum(r["Divergent Samples"] for r in results)

results.append({
    "Domain": "Overall",
    "Total Samples": overall_total,
    "Divergent Samples": overall_div,
    "% Divergent": round(overall_div / overall_total * 100, 2),
})

df = pd.DataFrame(results)

print(df)