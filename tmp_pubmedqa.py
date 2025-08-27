from datasets import load_dataset
import json
from collections import Counter


ds_l = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")  
print(f"Dataset size: {len(ds_l)}")

sample = ds_l[10]
print(json.dumps(sample, indent=2, ensure_ascii=False))


decisions = [ex["final_decision"] for ex in ds_l]
counter = Counter(decisions)

print("\nDecision counts:")
for k, v in counter.items():
    print(f"{k}: {v}")
