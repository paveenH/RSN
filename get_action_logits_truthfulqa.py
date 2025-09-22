#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthfulQA — action-style (0–9) willingness scoring
- For each sample and each role, read the logits for the final token over 0..9
- Compute softmax probabilities and argmax as the score
- Outputs are consistent with the MMLU-Pro action version:
  * JSON: {"data": [...(with score_* / score_prob_* / score_dist_* / logits_*)...],
           "summary": {role: counts/avg...}, "template": ...}
  * CSV : one row per task-role, with mean/std/entropy/tails/top_ratio/counts_0..9
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
import argparse
from tqdm import tqdm
import csv
import math

from llms import VicundaModel
from template import select_templates_pro
import utils



def main(args):

    # Load model
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load TruthfulQA multiple_choice JSON
    samples: List[Dict[str, Any]] = utils.load_json(TQA_PATH)
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError(f"Empty or invalid TQA JSON: {TQA_PATH}")

    # Task name & roles
    task_name = samples[0].get("task", "TruthfulQA")
    roles = utils.make_characters(task_name, "non")

    # Template: action (0..9)
    templates = select_templates_pro(suite="action")
    LABELS = templates["labels"]              # ["0","1",...,"9"]
    opt_ids = utils.option_token_ids(vc, LABELS)
    tmp_record = utils.record_template(roles, templates)

    # Per-role counters (0..9)
    role_stats = {r: {str(i): 0 for i in range(10)} | {"total": 0} for r in roles}

    all_outputs = []

    with torch.no_grad():
        for sample in tqdm(samples, desc=task_name):
            ctx = sample["text"]

            item_out = dict(sample)  # keep original fields
            for role in roles:
                prompt = utils.construct_prompt(vc, templates, ctx, role, False)

                # logits → take the last token
                logits = vc.get_logits([prompt], return_hidden=False)
                logits_np = logits[0, -1].detach().cpu().numpy()

                # keep only logits for 0..9
                opt_logits = np.array([logits_np[i] for i in opt_ids], dtype=float)

                # softmax
                opt_logits -= opt_logits.max()
                probs = np.exp(opt_logits)
                probs /= probs.sum()

                pred_idx = int(np.argmax(opt_logits))
                pred_label = LABELS[pred_idx]
                pred_prob = float(probs[pred_idx])

                key = role.replace(" ", "_")
                item_out[f"score_{key}"] = pred_label
                item_out[f"score_prob_{key}"] = pred_prob
                item_out[f"score_dist_{key}"] = probs.tolist()
                item_out[f"logits_{key}"] = opt_logits.tolist()

                # update counters
                rs = role_stats[role]
                rs["total"] += 1
                rs[pred_label] = rs.get(pred_label, 0) + 1

            all_outputs.append(item_out)

    # ===== Summary & print & write CSV =====
    rows = []
    for role, s in role_stats.items():
        total = int(s["total"])
        counts = [int(s[str(i)]) for i in range(10)]

        if total > 0:
            mean = sum(i * c for i, c in enumerate(counts)) / total
            var = sum(((i - mean) ** 2) * c for i, c in enumerate(counts)) / total
            std = math.sqrt(var)
            dist = np.array(counts, dtype=float) / total
            ent = utils.entropy_bits(dist)
            top_score = int(np.argmax(counts))
            top_ratio = max(counts) / total
            tail_low = sum(counts[0:3]) / total         # 0,1,2
            tail_high = sum(counts[8:10]) / total       # 8,9
        else:
            mean = std = ent = top_ratio = tail_low = tail_high = 0.0
            top_score = 0

        print(f"{role:<25} avg_score={mean:5.2f}  counts={counts}")

        # CSV row (aligned with MMLU-Pro action summary columns)
        row = {
            "model": args.model,
            "size": args.size,
            "suite": "action",
            "task": task_name,
            "role": role,
            "total": total,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "entropy_bits": round(ent, 6),
            "top_score": top_score,
            "top_ratio": round(top_ratio, 6),
            "tail_low_0_2": round(tail_low, 6),
            "tail_high_8_9": round(tail_high, 6),
        }
        for i in range(10):
            row[f"counts_{i}"] = counts[i]
        rows.append(row)

    # Save JSON (same format as MMLU-Pro action: includes data+summary+template)
    summary = {}
    for role, s in role_stats.items():
        total = int(s["total"])
        counts = [int(s[str(i)]) for i in range(10)]
        avg_score = (sum(i * c for i, c in enumerate(counts)) / total) if total > 0 else 0.0
        summary[role] = {**{str(i): counts[i] for i in range(10)},
                         "total": total, "avg_score": round(avg_score, 3)}

    ans_file = ANS_DIR / f"{task_name.replace(' ', '_')}_{args.model}_{args.size}_{args.mode}_action.json"
    utils.dump_json({"data": all_outputs, "summary": summary, "template": tmp_record}, ans_file)
    print("[Saved answers]", ans_file)

    # Save CSV (aligned with MMLU-Pro action summary columns)
    csv_file = ANS_DIR / f"summary_{args.model}_{args.size}_{args.mode}_action.csv"
    fieldnames = [
        "model","size","suite","task","role","total","mean","std","entropy_bits",
        "top_score","top_ratio","tail_low_0_2","tail_high_8_9"
    ] + [f"counts_{i}" for i in range(10)]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("[Saved summary]", csv_file)
    print("\n✅  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TruthfulQA action-style (0–9) willingness with VicundaModel")
    parser.add_argument("--mode", required=True, choices=["mc1", "mc2"], help="TruthfulQA mode (used only for naming)")
    parser.add_argument("--model", "-m", required=True, help="Model name, used for folder naming")
    parser.add_argument("--size", "-s", required=True, help="Model size, e.g., 8B")
    parser.add_argument("--model_dir", required=True, help="HF model id / local checkpoint directory")
    parser.add_argument("--ans_file", required=True, help="Subfolder name for outputs")
    args = parser.parse_args()

    # Prepare directories
    TQA_DIR = Path("/data2/paveen/RolePlaying/components/truthfulqa/")
    ANS_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}/")
    ANS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "mc1":
        TQA_PATH = TQA_DIR / "truthfulqa_mc1_validation_shuf.json"
    else:
        TQA_PATH = TQA_DIR / "truthfulqa_mc2_validation_shuf.json"

    print("Mode:", args.mode)
    print("Loading model from:", args.model_dir)
    print("Dataset:", TQA_PATH)

    main(args)