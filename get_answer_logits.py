#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
for every role on every MMLU task.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import torch
import argparse
import csv
from tqdm import tqdm
import h5py
from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates
import utils

# ─────────────────────────── Main ───────────────────────────────

def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    
    templates = select_templates(suite=args.suite, use_E=args.use_E, cot=args.cot)
    LABELS = templates["labels"]    

    rows = []  # collect stats for CSV
    for task in TASKS:
        print(f"\n=== {task} ===")
        opt_ids = utils.option_token_ids(vc, LABELS)
        data_path = MMLU_DIR / f"{task}.json"
        samples = utils.load_json(data_path)
        
        # Parse custom roles if provided
        custom_roles = None
        if args.roles:
            custom_roles = [r.strip() for r in args.roles.split(",")]
        roles = utils.make_characters(task, custom_roles)
        role_stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}

        tmp_record = utils.record_template(roles, templates)

        # Initialize HDF5 files for streaming save if needed
        h5_files: Dict[str, h5py.File] = {}
        h5_datasets: Dict[str, h5py.Dataset] = {}
        if args.save:
            for role in roles:
                safe_role = role.replace(" ", "_").replace("-", "_")
                hs_file = HS_DIR / f"{safe_role}_{task}_{args.size}.h5"
                h5_files[role] = h5py.File(hs_file, "w")
            sample_count = len(samples)

        with torch.no_grad():
            for sample_idx, sample in enumerate(tqdm(samples, desc=task)):
                ctx = sample["text"]
                true_idx = sample["label"]
                if not 0 <= true_idx < len(LABELS):
                    continue
                true_label = LABELS[true_idx]

                for role in roles:
                    prompt = utils.construct_prompt(vc, templates, ctx, role, args.use_chat)

                    logits = vc.get_logits([prompt], return_hidden=args.save)

                    if args.save and isinstance(logits, tuple):
                        logits, hidden = logits
                        last_hs = [lay[0, -1].half().cpu().numpy() for lay in hidden]  # FP16
                        hs_array = np.stack(last_hs, axis=0)  # (layers, hidden)
                        if role not in h5_datasets:
                            hs_shape = hs_array.shape
                            h5_datasets[role] = h5_files[role].create_dataset(
                                "hidden_states",
                                shape=(sample_count,) + hs_shape,
                                dtype="float16",
                                chunks=(1,) + hs_shape,
                            )
                        h5_datasets[role][sample_idx] = hs_array

                    logits = logits[0, -1].float().cpu().numpy()

                    # softmax over answer options
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

                    # update stats
                    rs = role_stats[role]
                    rs["total"] += 1
                    if pred_label == true_label:
                        rs["correct"] += 1
                    elif pred_label == "E":
                        rs["E_count"] += 1
                    else:
                        rs["invalid"] += 1

        # accuracy summary + collect CSV rows
        for role, s in role_stats.items():
            pct = s["correct"] / s["total"] * 100 if s["total"] else 0
            print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']}), E={s['E_count']}")
            rows.append({
                "model": args.model, "size": args.size, "suite": args.suite,
                "task": task, "role": role,
                "correct": s["correct"], "E_count": s["E_count"],
                "invalid": s["invalid"], "total": s["total"],
                "accuracy_percentage": round(pct, 2),
            })

        # save answers JSON
        ans_file = ANS_DIR / f"{task}_{args.size}_answers.json"
        utils.dump_json({"data": samples, "accuracy": {r: {**s, "accuracy_percentage": round(s["correct"]/s["total"]*100 if s["total"] else 0, 2)} for r, s in role_stats.items()}, "template": tmp_record}, ans_file)
        print("[Saved answers]", ans_file)

        # close HDF5 files
        if args.save:
            for role in roles:
                if role in h5_files:
                    h5_files[role].close()
                    safe_role = role.replace(" ", "_").replace("-", "_")
                    hs_file = HS_DIR / f"{safe_role}_{task}_{args.size}.h5"
                    print("[Saved HS]", hs_file)

    # save summary CSV
    csv_file = ANS_DIR / f"summary_{args.model}_{args.size}.csv"
    fieldnames = ["model", "size", "suite", "task", "role", "correct", "E_count", "invalid", "total", "accuracy_percentage"]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ Saved summary CSV → {csv_file}")

    print("\n All tasks finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run MMLU role-based extraction with hidden-state saving")
    parser.add_argument("--model", "-m", required=True, help="Model name, used for folder naming")
    parser.add_argument("--size", "-s", required=True, help="Model size, e.g., `8B`")
    parser.add_argument("--type", type=str, default="non", help="Role type identifier (deprecated, use --roles instead)")
    parser.add_argument("--model_dir", required=True, help="LLM checkpoint/model directory")
    parser.add_argument("--ans_file", required=True, help="LLM checkpoint/model directory")
    parser.add_argument("--suite", type=str, default="default", choices=["default","vanilla", "action"])
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--use_E", action="store_true", help="Use five-choice template (A–E); otherwise use four-choice (A–D)")
    parser.add_argument("--cot", action="store_true", help="Use cot template")
    parser.add_argument("--save", action="store_true", help="Whether to save hidden states (default saves only logits/answers)")
    parser.add_argument("--use_chat", action="store_true", help="Use tokenizer.apply_chat_template for prompts")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base directory for data/output (e.g., /work/<user>/RolePlaying/components). "
                             "If not set, falls back to /{data}/paveen/RolePlaying/components")
    parser.add_argument("--roles", type=str, default=None,
                        help="Comma-separated list of roles. Use {task} as placeholder for task name. "
                             "E.g., 'neutral,{task} expert,non {task} expert'")
    parser.add_argument("--hs_dir", type=str, default=None,
                        help="Base directory for hidden states output (e.g., /data1/paveen/ConfSteer/HiddenStates). "
                             "If not set, falls back to {base}/hidden_states_{type}/{model}/{task_name}/")
    parser.add_argument("--task_name", type=str, default="mmlu",
                        help="Task name for hidden states subdirectory (default: mmlu)")

    args = parser.parse_args()

    print("model: ", args.model)
    print("Loading model from:", args.model_dir)

    if args.base_dir:
        BASE = Path(args.base_dir)
    else:
        BASE = Path(f"/{args.data}/paveen/RolePlaying/components")

    MMLU_DIR = BASE / "mmlu"
    ANS_DIR = BASE / args.model / args.ans_file
    if args.hs_dir:
        HS_DIR = Path(args.hs_dir) / args.model / args.task_name
    else:
        HS_DIR = BASE / f"hidden_states_{args.type}" / args.model / args.task_name
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    HS_DIR.mkdir(parents=True, exist_ok=True)
    main()
