
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthfulQA (multiple_choice) runner for VicundaModel

- Supports MC1 (single-label) and MC2 (multi-label; correct if any gold label is predicted)
- Dynamically builds LABELS and option token ids per-question
- Saves per-sample predictions (answer/prob/softmax/logits) and per-role summary CSV

"""


from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
import argparse
from tqdm import tqdm
import csv

from llms import VicundaModel
from template import select_templates_pro
import utils


LETTER = [chr(ord("A") + i) for i in range(26)]  # A..Z

def labels_for_sample(sample: Dict[str, Any]) -> List[str]:

    K = max(1, min(len(sample["choices"]), len(LETTER)))
    return LETTER[:K]


def gold_indices_for_sample(sample: Dict[str, Any]) -> List[int]:
    """
    Get gold indices for a sample.
    """
    gi = sample.get("gold_indices")
    if gi and isinstance(gi, list) and len(gi) > 0:
        return [int(x) for x in gi]
    labels = sample.get("labels", [])
    pos = [i for i, v in enumerate(labels) if int(v) == 1]
    return pos if pos else [0]


def main(args):

    # Load model
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load TruthfulQA multiple_choice JSON
    samples: List[Dict[str, Any]] = utils.load_json(TQA_PATH)
    
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("Empty or invalid TQA JSON:", TQA_PATH)

    # Roles and stats
    task_name = samples[0].get("task", "TruthfulQA")
    roles = utils.make_characters()
    role_stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}

    all_outputs = []

    with torch.no_grad():
        for sample in tqdm(samples, desc=task_name):
            ctx = sample["text"]

            # Build per-question LABELS and templates
            LABELS = labels_for_sample(sample)
            templates = select_templates_pro(suite=args.suite, labels=LABELS, use_E=args.use_E)
            refusal_label = templates.get("refusal_label")

            opt_ids = utils.option_token_ids(vc, LABELS)
            gold_indices = gold_indices_for_sample(sample, args.mode)

            item_out = dict(sample)  # copy for output
            for role in roles:
                prompt = utils.construct_prompt(vc, templates, ctx, role, False)
                logits = vc.get_logits([prompt], return_hidden=False)
                logits_np = logits[0, -1].detach().cpu().numpy()

                opt_logits = np.array([logits_np[i] for i in opt_ids])
                probs = utils.softmax_1d(opt_logits)
                pred_idx = int(opt_logits.argmax())
                pred_label = LABELS[pred_idx]
                pred_prob = float(probs[pred_idx])

                key = role.replace(" ", "_")
                item_out[f"answer_{key}"] = pred_label
                item_out[f"prob_{key}"] = pred_prob
                item_out[f"softmax_{key}"] = probs.tolist()
                item_out[f"logits_{key}"] = opt_logits.tolist()

                rs = role_stats[role]
                rs["total"] += 1
                if pred_idx in gold_indices:
                    rs["correct"] += 1
                elif args.use_E and refusal_label is not None and pred_label == refusal_label:
                    rs["E_count"] += 1
                else:
                    rs["invalid"] += 1

            all_outputs.append(item_out)

    # Print and summarize
    rows = []
    for role, s in role_stats.items():
        acc = (s["correct"] / s["total"] * 100.0) if s["total"] else 0.0
        print(f"{role:<25} acc={acc:5.2f}%  (correct {s['correct']}/{s['total']}), Refuse={s['E_count']}")
        rows.append({
            "model": args.model,
            "size": args.size,
            "dataset": "TruthfulQA",
            "mode": args.mode.upper(),
            "task": task_name,
            "role": role,
            "correct": s["correct"],
            "E_count": s["E_count"],
            "invalid": s["invalid"],
            "total": s["total"],
            "accuracy_percentage": round(acc, 2),
            "suite": args.suite,
            "refusal_enabled": int(bool(args.use_E)),
            "refusal_label": refusal_label if refusal_label is not None else "",
        })

    # Save answers (full)
    tmp_record = utils.record_template(roles, templates)
    ans_file = ANS_DIR / f"{task_name.replace(' ', '_')}_{args.model}_{args.size}_{args.mode}.json"
    utils.dump_json({"data": all_outputs, "template": tmp_record}, ans_file)
    print("[Saved answers]", ans_file)

    # Save CSV summary
    csv_file = ANS_DIR / f"summary_{args.model}_{args.size}_{args.mode}.csv"
    fieldnames = [
        "model", "size", "dataset", "mode", "task", "role",
        "correct", "E_count", "invalid", "total",
        "accuracy_percentage", "suite", "refusal_enabled", "refusal_label"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("[Saved summary]", csv_file)
    print("\n✅  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TruthfulQA MC1/MC2 with VicundaModel")
    parser.add_argument("--mode", required=True, choices=["mc1", "mc2"], help="TruthfulQA mode")
    parser.add_argument("--model", "-m", required=True, help="Model name, used for folder naming")
    parser.add_argument("--size", "-s", required=True, help="Model size, e.g., 8B")
    parser.add_argument("--model_dir", required=True, help="HF model id / local checkpoint dir")
    parser.add_argument("--ans_file", required=True, help="Subfolder name for outputs")
    parser.add_argument("--use_E", action="store_true", help="Enable 5-choice template (if template requires E option)")
    parser.add_argument("--suite", type=str, default="default", choices=["default", "vanilla"], help="Prompt suite name")
    args = parser.parse_args()
    
    # Prepare directories
    TQA_DIR = Path("/data2/paveen/RolePlaying/components/truthfulqa/")
    ANS_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}/")
    
    if args.mode == "mc1":
        TQA_PATH = TQA_DIR / "truthfulqa_mc1_validation.json"
    else:
        TQA_PATH = TQA_DIR / "truthfulqa_mc2_validation.json"
    

    print("Mode:", args.mode)
    print("Loading model from:", args.model_dir)
    
    main(args)
    
        