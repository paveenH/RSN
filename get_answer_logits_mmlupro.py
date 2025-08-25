#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 17:27:15 2025

Extract highest-logit answer + probability **and** save last-token hidden states
for every role on every task — switched to MMLU-Pro combined JSON.

@author: paveenhuang

"""

from pathlib import Path
from typing import List
import numpy as np
import torch
import argparse
from tqdm import tqdm
import csv

from llms import VicundaModel
from template import select_templates_pro
import utils


def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load mmlupro json file
    all_samples: List[dict] = utils.load_json(MMLU_PRO_DIR)

    # group by "task"
    tasks = sorted({s["task"] for s in all_samples})
    print(f"Found {len(tasks)} tasks in MMLU-Pro JSON.")
        
    rows = []  # collect stats for CSV
    for task in tasks:
        print(f"\n=== {task} ===")
        samples = [s for s in all_samples if s["task"] == task]
        if not samples:
            raise ValueError("empty task:", task)
            
            
        # labels
        max_label = max(int(s["label"]) for s in samples)
        K = max(1, max_label + 1)
        labels = [chr(ord("A") + i) for i in range(K)]
        print(labels)
        
        templates = select_templates_pro(suite=args.suite, labels=labels, use_E=args.use_E)
        LABELS = templates["labels"]
        print(LABELS)
        refusal_label = templates.get("refusal_label")
        print ("refuse label ", refusal_label)

        # get ids of options
        opt_ids = utils.option_token_ids(vc, LABELS)

        # role list
        roles = utils.make_characters(task.replace(" ", "_"), args.type)
        role_stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}

        tmp_record = utils.record_template(roles, templates)

        with torch.no_grad():
            for sample in tqdm(samples, desc=task):
                ctx = sample["text"]
                true_idx = int(sample["label"])
                if not 0 <= true_idx < len(LABELS):
                    raise ValueError(
                        f"[Error] Task {task} has invalid label index {true_idx} "
                        f"(valid range: 0–{len(LABELS)-1}). Sample: {sample}"
                    )

                true_label = LABELS[true_idx]

                for role in roles:
                    prompt = utils.construct_prompt(vc, templates, ctx, role, False)
                    logits = vc.get_logits([prompt], return_hidden=False)
                    logits = logits[0, -1].cpu().numpy()

                    # Only in k options in the task
                    opt_logits = np.array([logits[i] for i in opt_ids])
                    probs = utils.softmax_1d(opt_logits)
                    pred_idx = int(opt_logits.argmax())
                    pred_label = LABELS[pred_idx]
                    pred_prob = float(probs[pred_idx])

                    # attach answer+prob to sample
                    sample[f"answer_{role.replace(' ', '_')}"] = pred_label
                    sample[f"prob_{role.replace(' ', '_')}"] = pred_prob
                    sample[f"softmax_{role.replace(' ', '_')}"] = probs.tolist()
                    sample[f"logits_{role.replace(' ', '_')}"] = opt_logits.tolist()

                    # statistics
                    rs = role_stats[role]
                    rs["total"] += 1
                    
                    if pred_label == true_label:
                        rs["correct"] += 1
                    elif args.use_E and pred_label == refusal_label:
                        rs["E_count"] += 1
                    else:
                        rs["invalid"] += 1

        # summary
        for role, s in role_stats.items():
            pct = s["correct"] / s["total"] * 100 if s["total"] else 0
            print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']}), Refuse={s['E_count']}")
            rows.append({
                "model": args.model,
                "size": args.size,
                "task": task,
                "role": role,
                "correct": s["correct"],
                "E_count": s["E_count"],
                "invalid": s["invalid"],
                "total": s["total"],
                "accuracy_percentage": s["accuracy_percentage"],
                "suite": args.suite,
                "refusal_enabled": int(bool(args.use_E)),
                "refusal_label": refusal_label if refusal_label is not None else "",
            })
        
        
        # save
        task_dir = ANS_DIR / f"{args.model}"
        task_dir.mkdir(parents=True, exist_ok=True)
        ans_file = task_dir / f"{task.replace(' ', '_')}_{args.size}_answers.json"
        utils.dump_json({"data": samples, "template": tmp_record}, ans_file)
        print("[Saved answers]", ans_file)

    # save task performance
    csv_file = ANS_DIR / f"summary_{args.model}_{args.size}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "role", "correct", "E_count", "total"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMLU-Pro role-based extraction with hidden-state saving")
    parser.add_argument("--model", "-m", required=True, help="Model name, used for folder naming")
    parser.add_argument("--size", "-s", required=True, help="Model size, e.g., `8B`")
    parser.add_argument("--type", required=True, help="Role type identifier, affects prompt and output directories")
    parser.add_argument("--model_dir", required=True, help="LLM checkpoint/model directory")
    parser.add_argument("--ans_file", required=True, help="Subfolder name for outputs")
    parser.add_argument("--use_E", action="store_true", help="Use 5-choice template (A–E)")
    parser.add_argument("--suite", type=str, default="default", choices=["default", "vanilla"], help="Prompt suite for MMLU-Pro")
    
    args = parser.parse_args()

    print("model: ", args.model)
    print("Loading model from:", args.model_dir)

    MMLU_PRO_DIR = Path("/data2/paveen/RolePlaying/components/mmlupro/mmlupro_test.json")
    ANS_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}/")
    ANS_DIR.mkdir(parents=True, exist_ok=True)

    main()
