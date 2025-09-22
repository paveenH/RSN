#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

file1 = Path("/data2/paveen/RolePlaying/components/truthfulqa/truthfulqa_mc1_validation_shuf.json")
file2 = Path("/data2/paveen/RolePlaying/components/truthfulqa/truthfulqa_mc2_validation_shuf.json")

out_file = Path("/data2/paveen/RolePlaying/components/truthfulqa/truthfulqa_mc1_mc2_merged.json")

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    data1 = load_json(file1)
    data2 = load_json(file2)

    if not isinstance(data1, list) or not isinstance(data2, list):
        raise ValueError("Both input files must contain JSON lists.")

    merged = data1 + data2
    print(f"Loaded {len(data1)} samples from {file1.name}")
    print(f"Loaded {len(data2)} samples from {file2.name}")
    print(f"Total merged: {len(merged)}")

    save_json(merged, out_file)
    print(f"✅ Saved merged file → {out_file}")

if __name__ == "__main__":
    main()