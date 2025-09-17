#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run action-style scoring (0–9) on a flat MMLU/MMLU-Pro JSON and export:

"""

from pathlib import Path
from typing import List
import numpy as np
import torch
import argparse
from tqdm import tqdm
import csv
import math

from llms import VicundaModel
from template import select_templates_pro
import utils


def _entropy_bits(p: np.ndarray) -> float:
    """Shannon entropy in bits"""
    p = np.asarray(p, dtype=float)
    p = p[(p > 0) & np.isfinite(p)]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load json files
    all_samples: List[dict] = utils.load_json(DATA_DIR)
    tasks = sorted({s["task"] for s in all_samples})
    print(f"Found {len(tasks)} tasks.")

    # template and label
    templates = select_templates_pro(suite="action")
    LABELS = templates["labels"]                       # ["0",...,"9"]
    opt_ids = utils.option_token_ids(vc, LABELS)       #  token id

    rows = []  
    for task in tasks:
        print(f"\n=== {task} ===")
        samples = [s for s in all_samples if s["task"] == task]
        roles = utils.make_characters(task.replace(" ", "_"), args.type)
        role_stats = {r: {str(i): 0 for i in range(10)} | {"total": 0} for r in roles}
        tmp_record = utils.record_template(roles, templates)

        with torch.no_grad():
            for sample in tqdm(samples, desc=task):
                ctx = sample["text"]

                for role in roles:
                    prompt = utils.construct_prompt(vc, templates, ctx, role, False)

                    logits = vc.get_logits([prompt], return_hidden=False)
                    logits = logits[0, -1].detach().cpu().numpy()

                    opt_logits = np.array([logits[i] for i in opt_ids], dtype=float)
                    # softmax
                    opt_logits -= opt_logits.max()         
                    probs = np.exp(opt_logits)
                    probs /= probs.sum()

                    pred_idx = int(np.argmax(opt_logits))
                    pred_label = LABELS[pred_idx]           # "0"..."9"
                    pred_prob = float(probs[pred_idx])

                    rk = role.replace(" ", "_")
                    sample[f"score_{rk}"] = pred_label
                    sample[f"score_prob_{rk}"] = pred_prob
                    sample[f"score_dist_{rk}"] = probs.tolist()
                    sample[f"logits_{rk}"] = opt_logits.tolist()

                    # Statistic
                    rs = role_stats[role]
                    rs["total"] += 1
                    rs[pred_label] = rs.get(pred_label, 0) + 1

        # === Wtite to csv, and print per-task  ===
        task_dir = ANS_DIR / f"{args.model}"
        task_dir.mkdir(parents=True, exist_ok=True)
        ans_file = task_dir / f"{task.replace(' ', '_')}_{args.size}_answers.json"

        # summary
        summary = {}
        for role, s in role_stats.items():
            total = int(s["total"])
            counts = [int(s[str(i)]) for i in range(10)]
            if total > 0:
                mean = sum(i * c for i, c in enumerate(counts)) / total
                var = sum(((i - mean) ** 2) * c for i, c in enumerate(counts)) / total
                std = math.sqrt(var)
                dist = np.array(counts, dtype=float) / total
                ent = _entropy_bits(dist)
                top_score = int(np.argmax(counts))
                top_ratio = max(counts) / total
                tail_low = sum(counts[0:3]) / total         # 0,1,2
                tail_high = sum(counts[8:10]) / total       # 8,9
            else:
                mean = std = ent = top_ratio = tail_low = tail_high = 0.0
                top_score = 0

            # per-role summary
            summary[role] = {
                **{str(i): counts[i] for i in range(10)},
                "total": total,
                "avg_score": round(mean, 3),
            }

            # CSV row
            row = {
                "model": args.model,
                "size": args.size,
                "suite": "action",
                "task": task,
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
            
            print(f"{role:<25} mean={mean:.2f}, std={std:.2f}, ent={ent:.2f}, "
              f"top_score={top_score}, top_ratio={top_ratio:.2f}, "
              f"tail_low={tail_low:.2f}, tail_high={tail_high:.2f}")

        # Save
        utils.dump_json({"data": samples, "summary": summary, "template": tmp_record}, ans_file)
        print("[Saved answers]", ans_file)

    # === Write CSV ===
    csv_file = ANS_DIR / f"summary_{args.model}_{args.size}.csv"
    fieldnames = [
        "model","size","suite","task","role","total","mean","std","entropy_bits",
        "top_score","top_ratio","tail_low_0_2","tail_high_8_9",
    ] + [f"counts_{i}" for i in range(10)]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved summary CSV to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run action-style (0–9) role-based extraction")
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--size", "-s", required=True)
    parser.add_argument("--type", required=True, help="role family, e.g. non/expert/... used by make_characters")
    parser.add_argument("--test_file", required=True, help="flat JSON with fields: task, text, (label)")
    parser.add_argument("--ans_file", required=True, help="output subfolder name under components/")
    args = parser.parse_args()

    print("model:", args.model)
    print("Loading model from:", args.model_dir)

    DATA_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.test_file}")
    ANS_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}/")
    ANS_DIR.mkdir(parents=True, exist_ok=True)

    main()